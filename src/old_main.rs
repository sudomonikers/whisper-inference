use candle::{DType, Device, Tensor};
use candle::utils::{cuda_is_available, metal_is_available};
use clap::Parser;
use tokenizers::Tokenizer;
use candle_nn::VarBuilder;
use anyhow::{Error as E, Result};

#[derive(Parser, Debug)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model id of the model on huggingface
    #[clap(long, conflicts_with = "model_path")]
    model_id: Option<String>,

    /// Path to model weights, in safetensors format.
    #[clap(long, conflicts_with = "model_id")]
    model_path: Option<String>,
}

fn load_model_from_local_path(model_path: &Option<String>) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    match model_path {
        Some(path) => Ok(std::path::PathBuf::from(path)),
        None => Err("Model path provided does not match a safetensors file. Check your path provided and try again.".into()),
    }
}

// fn load_model_from_hugging_face(model_id: &Option<String>) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
//     let api = hf_hub::api::sync::Api::new()?;
//     let model_id = model_id.as_ref().ok_or_else(|| "Model ID is required")?;
//     let api = api.model(model_id.to_string());

//     Ok(api.get(&format!("yolov8{size}{task}.safetensors")))
// }

pub fn load_model(model_id: &Option<String>, model_path: &Option<String>) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    match (model_id.is_some(), model_path.is_some()) {
        (true, false) => {
            // Model ID is provided, handle accordingly
            println!("\nModel ID provided: {:?}\n", model_id);
            //load_model_from_hugging_face(model_id)
            Err("Retrieving a model from hugging face not yet implemented".into())
        }
        (false, true) => {
            // Model Path is provided, handle accordingly
            println!("\nModel Path provided: {:?}\n", model_path);
            load_model_from_local_path(model_path)
        }
        _ => {
            // Display an error message and exit if neither or both are provided
            eprintln!("\nError: Please provide exactly one of --model_id or --model_path\n");
            std::process::exit(1);
        }
    }
}

pub fn device(cpu: bool) -> Result<Device, Box<dyn std::error::Error>> {
    if cpu {
        print!("\n--cpu command line argument specified. Running on CPU\n");
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        print!("\nRunning on CUDA\n");
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        print!("\nRunning on METAL\n");
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "\nRunning on CPU, to run on GPU(metal), build this example with `--features metal`\n"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("\nRunning on CPU, to run on GPU, build this example with `--features cuda`\n");
        }
        Ok(Device::Cpu)
    }
}

pub fn load_safetensors(json_file: &str) -> Result<Vec<std::path::PathBuf>> {
    let json_file = std::fs::File::open("models/yi-6b/tokenizer.json")?;
    let json: serde_json::Value = serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("\nCommand Line Arguments: {:?}\n", args);
    let device = device(args.cpu).unwrap();

    let tokenizer_file = std::path::PathBuf::from("models/yi-6b/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let safe_tensor_files_index = std::path::PathBuf::from("models/yi-6b/model.safetensors.index.json");
    let safe_tensor_files_paths = load_safetensors(json_file);
    let safe_tensors = unsafe { VarBuilder::from_mmaped_safetensors(&safe_tensor_files_paths, dtype, &device)? };


    let model = load_model(&args.model_id, &args.model_path);
    print!("\nModel Loaded Successfully: {:?}\n", model);

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}