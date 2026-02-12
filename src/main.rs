use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM as Qwen3};
use std::io::Write;
use tokenizers::Tokenizer;


struct TextGeneration {
    model: Qwen3,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    fn new(
        model: Qwen3,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<usize> {
        use std::io::Write;
        println!("开始处理提示词...");
        
        
        let tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        
        let mut all_tokens = tokens.clone();
        let mut generated_text = 0usize;

        println!("生成中: ");
        std::io::stdout().flush()?;

        for index in 0..sample_len {
            let context_size: usize = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let ctxt = &all_tokens[start_pos..];
            
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            generated_text += 1;
            
            // 解码并打印新生成的 token
            if let Ok(text) = self.tokenizer.decode(&[next_token], false) {

                if text == "<endoftext>" || text == "<|im_end|>" {
                    println!("\n=== 生成结束 ===");
                    break;
                }
                print!("{}", text);
                std::io::stdout().flush()?;
                
                //generated_text.push_str(&text);
            }

        }
        
        println!("\n");
        Ok(generated_text)
    }
}

fn main() -> Result<()> {
    println!("=== Qwen3 Candle 对话系统 ===\n");

    // 设置模型路径 - 请修改为你的本地路径
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "E:\\projests\\alltest\\rust\\qwenthinking\\model\\qwen_06".to_string());
    
    println!("模型路径: {}", model_path);
    println!("正在初始化设备...");

    // 选择设备
    let device = Device::cuda_if_available(0)?;
    println!("使用设备: {:?}", device);

    // 加载 tokenizer
    println!("正在加载 tokenizer...");
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    println!("Tokenizer 路径: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    println!("Tokenizer 加载成功！");

    // 加载配置
    println!("正在加载模型配置...");
    let config_path = format!("{}/config.json", model_path);
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    println!("配置加载成功！");

    // 加载模型权重
    println!("正在加载模型权重...");
    let weights_path01 = format!("{}/model.safetensors", model_path);
    // let weights_path02 = format!("{}/model-00002-of-00003.safetensors", model_path);
    // let weights_path03 = format!("{}/model-00003-of-00003.safetensors", model_path);
    
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path01], DType::F32, &device)?
    };
    
    let model = Qwen3::new(&config, vb)?;
    println!("模型加载成功！\n");

    // 创建文本生成器
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        299792458, // seed
        Some(0.7),  // temperature
        Some(0.9),  // top_p
        1.1,        // repeat_penalty
        64,         // repeat_last_n
        &device,
    );

    println!("初始化完成！开始对话...\n");
    println!("输入 'quit' 或 'exit' 退出程序\n");

    // 交互式对话循环
    loop {
        print!("用户: ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("再见！");
            break;
        }

        // 构建对话格式的提示词
        let prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", input);

        print!("助手: ");
        std::io::stdout().flush()?;

        // 生成回复
        match pipeline.run(&prompt, 10000) {
            Ok(_) => {},
            Err(e) => eprintln!("生成错误: {}", e),
        }
    }

    Ok(())
}