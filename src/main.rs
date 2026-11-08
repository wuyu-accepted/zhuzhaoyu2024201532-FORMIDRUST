//从这里开始写。先读 README.md，再看 tests/public.rs 里测了什么。
//
// 规则：不许让 AI 写代码，可以问 AI 解释报错。

//引入一些库
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
type Pair = [u32; 2];
#[derive(Debug, Serialize, Deserialize)]
struct Model {
    merges: Vec<Pair>,
}
//&str 是 UTF-8 文本的共享借用；返回一个拥有数据的新 ID 序列。
fn bytes_to_ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
//在一串 token ID 中，把指定的相邻 token 对替换成一个新的 ID，并返回替换后的新序列。
fn merge_pair(ids:&[u32],pair:Pair,new_id:u32)->Vec<u32>{
    let mut result = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i<ids.len(){
        if i<ids.len()-1 && ids[i]==pair[0] && ids[i+1]==pair[1]{
            //匹配的一对已经同时消费，前进两格
            result.push(new_id);
            i+=2;
        }else{
            //不匹配或只剩一个 token，原样保留；任何输入 token 都恰好被消费一次。
            result.push(ids[i]);
            i+=1;
        }
    }
    result
}
//为 Model 定义方法和关联函数。
impl Model{
    // 第 rank 条规则产生 ID 256 + rank，输入只能引用更小的 ID。
    fn validate(&self)->Result<(),String>{
        for (rank, pair) in self.merges.iter().enumerate(){
            // pair 是 &[u32; 2]，分别取出两个 ID。
            let left = pair[0];
            let right = pair[1];
            // 第一步：计算新 token 的 ID，检查 usize 加法是否溢出。
            let id: usize = match rank.checked_add(256) {
                Some(value) => value,
                None => {
                    return Err("模型词表超出 u32 ID 范围".to_string());
                }
            };
            // 第二步：将 usize 转换为 u32，检查数值能否装入 u32。
            let new_id: u32 = match u32::try_from(id) {
                Ok(value) => value,
                Err(_) => {
                    return Err("模型词表超出 u32 ID 范围".to_string());
                }
            };
            // 第三步：检查 left 和 right 是否小于 new_id。
            if left >= new_id || right >= new_id {
                return Err(format!("第 {} 条合并规则 [{left}, {right}] 引用了尚不存在的 ID（必须小于 {new_id}）",rank + 1));
            }
        }
        Ok(())
    }
    //返回(训练模型, 训练语料最终 token 数)
    fn train(text:&str,vocab_size:usize)->Result<(Self, usize), String>{
         if vocab_size < 256 {
            return Err("--vocab-size 不能小于 256".into());
        }
        let mut ids=bytes_to_ids(text);
        let mut merges = Vec::new();
        while merges.len() < vocab_size-256{
            let mut counts: HashMap<Pair, usize> = HashMap::new();
            // windows 每次前进一个位置，统计允许重叠。
            for window in ids.windows(2) {
                //* 解引用，修改计数本身；不是给引用变量加一。
                let pair = [window[0], window[1]];
                let count: &mut usize = counts.entry(pair).or_insert(0);
                *count += 1;
            }
            //None 则尚未选出候选；Some((pair,count)) 就保存当前最佳对及频次。
            let mut best: Option<(Pair, usize)> = None;
            for (pair, count) in counts{
                //循环遍历 counts 中的每个 (pair, count)，用 replace 判断是否要更新最佳候选
                let replace=match best{
                    None=>true,
                    Some((best_pair, best_count))=>{
                        count > best_count || (count == best_count && pair < best_pair)
                    }
                };
                if replace {
                    best = Some((pair, count));
                }
            }
            //取出最佳候选
            let (pair, count) = match best {
                Some((pair, count)) => (pair, count),
                None => {
                    break;
                }
            };
            //最高频次不足 2 时不再学习
            if count < 2 {
                break;
            }
            let merge_count: usize = merges.len();
            // 新 ID = 256 + 规则数量，先检查 usize 加法是否溢出。
            let id: usize = match merge_count.checked_add(256) {
                Some(value) => value,
                None => {
                    return Err("训练生成的 ID 超出 u32 范围".to_string());
                }
            };  
            // 再检查这个值能否转换为 u32。
            let new_id: u32 = match u32::try_from(id) {
                Ok(value) => value,
                Err(_) => {
                    return Err("训练生成的 ID 超出 u32 范围".to_string());
                }
            };
            ids = merge_pair(&ids, pair, new_id);
            merges.push(pair);
        }
        Ok((Self { merges }, ids.len()))
    }
    fn encode (&self,text:&str)->Result<Vec<u32>,String>{
        self.validate()?;
        let mut ranks=HashMap::new();
        //把合并规则整理成一个查找表：每个相邻 token 对对应它第一次出现的规则序号。
        for (rank,&pair) in self.merges.iter().enumerate(){
            match ranks.entry(pair) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    // pair 已经记录过，保留原来的 rank。
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    // pair 还没有记录，将当前 rank 存进去。
                    entry.insert(rank);
                }
            }
        }
        let mut ids=bytes_to_ids(text);
        loop{
            let mut best: Option<(usize, Pair)> = None;
            for window in ids.windows(2){
                let pair = [window[0], window[1]];
                // Some(&rank) 表示这对 token 有对应的合并规则；
                // None 表示模型里没有这条规则。
                match ranks.get(&pair){
                    Some(rank_reference) =>{
                        // 复制出 rank 的数值，千万不要把对 ranks 的引用存进 best。
                        let rank = *rank_reference;
                        match best{
                            None => {
                                // 这是遇到的第一个可合并相邻对，先作为当前最佳候选。
                                best = Some((rank, pair));
                            }
                            Some((best_rank, _best_pair)) => {
                            // rank 越小，规则在模型中的顺序越靠前，优先级越高。
                                if rank < best_rank {
                                    best = Some((rank, pair));
                                }
                            }
                        }
                    }
                    None => {
                        // 当前相邻对没有合并规则，跳过它，继续检查下一个窗口。
                    }
                }
            }
            let Some((rank, pair)) = best else {
                break;
            };
            let new_id = u32::try_from(rank + 256).map_err(|error| format!("ID 转换失败：{error}"))?;
            ids = merge_pair(&ids, pair, new_id);
        }
        Ok(ids)
    }
    //用显式栈展开 token，避免递归调用栈随规则深度增长。
    fn append_bytes(&self, id: u32, output: &mut Vec<u8>) -> Result<(), String>{
        let mut stack = vec![id];
        while let Some(current)=stack.pop(){
            if current<256{
                output.push(current as u8);
            }else{
                 // 新 token ID 从 256 开始，减去 256 得到它在 merges 中的下标。
                let index_result = usize::try_from(current - 256);
                let index: usize = match index_result {
                    Ok(value) => value,
                    Err(_) => {
                        return Err(format!("不存在的 token ID：{current}"));
                    }
                };
                // 确保get不会发生越界panic错误：找到规则就返回Some，找不到就返回None。
                let pair_option = self.merges.get(index);
                let pair_reference: &[u32; 2] = match pair_option {
                    Some(pair) => pair,
                    None => {
                    return Err(format!("不存在的 token ID：{current}"));
                    }
                };
                let left: u32 = pair_reference[0];
                let right: u32 = pair_reference[1];
                stack.push(right);
                stack.push(left);
            }
        }
        Ok(())
    }
    //借用输入 ID 序列，按顺序展开成字节；未知 ID 或最终 UTF-8 非法都返回 Err。
    fn decode(&self,ids:&[u32])->Result<String, String>{
        self.validate()?;
        let mut bytes=Vec::new();
        for &id in ids{
            self.append_bytes(id,&mut bytes)?;
        }
        let text: String = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => {
            return Err(format!("解码结果不是合法 UTF-8：{error}"));
            }
        };
        Ok(text)
    }
    fn vocab(&self, top: usize) -> Result<Vec<(usize, Vec<u8>)>, String>{
        self.validate()?;
        //饱和减法：top 超过规则数量时结果为 0，而不是无符号下溢。
        let start = self.merges.len().saturating_sub(top);
        let mut result = Vec::new();
        //.. 表示左闭右开区间；按 rank 从小到大显示选中的最后若干规则。
        for rank in start..self.merges.len() {
            let id = u32::try_from(rank + 256).map_err(|_| "模型词表超出 u32 ID 范围")?;
            let mut bytes = Vec::new();
            self.append_bytes(id, &mut bytes)?;
            result.push((rank + 256, bytes));
        }
        Ok(result)
    }
}
//引入标准库和命令行解析工具
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use clap::{Parser, Subcommand};

//derive 在编译时为 Cli 自动生成 Parser 等相关实现。
#[derive(Parser)]
#[command(name = "mini-bpe", version, about = "我的字节级 BPE 分词器")]
//一次完整的命令行输入
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
//自动把枚举变体映射成子命令
#[derive(Subcommand)]
//命令类型
enum Commands {
    //解析训练命令
    Train{
        corpus: PathBuf,
        //long 将 vocab_size 转为 --vocab-size；default_value_t 使用有类型的默认值 512。
        #[arg(long, default_value_t = 512)]
        vocab_size: usize,
        //short、long 根据字段名生成 -o、--output；未指定时保存到 model.json。
        #[arg(short, long, default_value = "model.json")]
        output: PathBuf,
    },
    //解析编码命令
    Encode{  
        #[arg(short, long)]
        model: PathBuf,
        //允许位置文本以 - 开头，例如 -hello，不把它直接误认为选项。
        //Some(text) 表示提供了文本；None 表示未提供，需要读标准输入。
        #[arg(allow_hyphen_values = true)]
        text: Option<String>,
    },
    //解析解码命令
    Decode{
        //根据下方 model 字段生成 -m 和 --model
        #[arg(short, long)]
        model: PathBuf,
        ids: Vec<u32>,
    },
    //解析词表命令
    Vocab{
        //根据下方 model 字段生成 -m 和 --model；无默认值的非 Option 字段必须提供。
        #[arg(short, long)]
        model: PathBuf,
        //默认查看最近学到的 10 个新 token
        #[arg(long="top", default_value_t = 10)]
        top_k: usize,
    },
}
//读取模型文件
fn load_model(path:&Path)->Result<Model,String>{
    //读取文件内容；成功就保存到 content，失败就生成易读的错误信息，并提前结束当前函数。
    let content:String=fs::read_to_string(path).map_err(|error| format!("读取模型 {} 失败：{error}", path.display()))?;
    //反序列化：把 JSON 字符串转换为 Model。
    let model: Model = serde_json::from_str(&content).map_err(|error| format!("模型 {} 格式不正确：{error}", path.display()))?;
    //进一步验证规则只能引用已经存在的 ID；map_err 加上模型路径，便于定位文件。
    model.validate().map_err(|error| format!("模型 {} 无效：{error}", path.display()))?;
    Ok(model)
}
//执行命令
fn run(command:Commands)->Result<String,String>{
    match command{
        Commands::Train{corpus,vocab_size,output} => {
            //基础检查：词表大小不能小于 256
            if vocab_size<256{
                return Err("--vocab-size 不能小于 256".into());
            }
            //借用路径读取语料，不移动 corpus，便于错误信息继续使用 corpus.display()。
            let text=fs::read_to_string(&corpus).map_err(|error| format!("读取语料 {} 失败：{error}", corpus.display()))?;
            let (model,tokens)=Model::train(&text,vocab_size)?;
            //as f64 使除法使用浮点运算。
            let percent=if text.is_empty(){0.0}else{tokens as f64*100.0/text.len() as f64};
            let json=serde_json::to_string(&model).map_err(|error| format!("序列化模型失败：{error}"))?;
            //将 JSON 写到目标文件
            fs::write(&output,json).map_err(|error| format!("写模型 {} 失败：{error}", output.display()))?;
            Ok(format!("合并 {} 次，词表 {}；语料 {} 字节 → {} 个 token(压缩到 {:.2}%)\n",model.merges.len(),256 + model.merges.len(),text.len(),tokens,percent))
        }
        Commands::Encode{model,text}=>{
            let model = load_model(&model)?;
            let text=match text{
                //这里把 Option 内的 String 移出并交给新绑定，不需要 clone。
                Some(text)=>text,
                //用独占的可变借用&mut，让读取函数修改缓冲区
                None=>{
                    let mut buffer=String::new();
                    io::stdin().read_to_string(&mut buffer).map_err(|error| format!("读取标准输入失败：{error}"))?;
                    buffer
                }
            };
            let ids=model.encode(&text)?;
            //iter 借用遍历 ID，然后用 map 把每个 ID 变成十进制 String，join 用一个空格连接
            Ok(format!("{}\n",ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")))
        }
        Commands::Decode { model, ids } => {
            load_model(&model)?.decode(&ids)
        }
        Commands::Vocab{model,top_k}=>{
            let mut output=String::new();
            //依次接收vocab的(ID, 字节串)
            for(id,bytes) in load_model(&model)?.vocab(top_k)?{
                //不是每个 token 都是完整的汉字，所以用String::from_utf8_lossy
                output.push_str(&format!("{id}\t{:?}\t{bytes:?}\n",String::from_utf8_lossy(&bytes)));
            }
            Ok(output)
        }
    }
}
//打印错误信息到标准错误，并返回非零退出码
fn report_error(message: &str) -> ExitCode {
    let _ = writeln!(io::stderr().lock(), "mini-bpe: {message}");
    ExitCode::from(1)
}
fn main() -> ExitCode {
    //读取进程参数并解析。try_parse 返回 Result，允许我自定错误退出码。
    let cli:Cli = match Cli::try_parse(){
        Ok(cli) => cli,
        Err(error) => {
            let failed=error.use_stderr();
            let _=error.print();
            //让 clap 按自身格式打印帮助、版本或参数错误；忽略打印失败的返回值。
            return if failed{
                ExitCode::from(1)
            }else{
                ExitCode::SUCCESS
            };
        }
    };
    match run(cli.command){
        Ok(output) => match io::stdout().lock().write_all(output.as_bytes()) {
            //嵌套 match 分别处理业务成功与输出是否成功，输出失败也返回 1
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => report_error(&format!("写入标准输出失败：{error}")),
        },
        Err(error) => report_error(&error),
    }
}

