import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

HF_TOKEN = "<TOKKEN>"
login(HF_TOKEN)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    use_fast=True,
    padding_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

file_path = "testD.csv"
data_sheet_details = pd.read_csv(file_path).reset_index(drop=True)

k = 8  
num_batches_to_run = 900  

batches = []
for i in range(0, len(data_sheet_details) - (k + 1), k + 1):
    batch = data_sheet_details.iloc[i:i + k]
    test_case = data_sheet_details.iloc[i + k]
    batches.append((batch, test_case))

batches = batches[:num_batches_to_run]

results = []

def extract_emissions(text):
    prediction_pattern = r"Rail-GHG-Emissions-Tonnes:\s*([\d.]+)\s*\|\s*Truck-GHG-Emissions-Tonnes:\s*([\d.]+)"
    matches = re.findall(prediction_pattern, text)
    if matches:
        rail_value, truck_value = matches[-1]
        return float(rail_value), float(truck_value)
    return None, None

for batch_num, (batch, test_case) in enumerate(batches):
    k_shot_examples = [
        {
            "feature": (
                f"Date: {row['Waybill Date']} | Origin: {row['Origin']} ({row['Origin-St']}) | "
                f"Destination: {row['Destination']} ({row['Destination-St']}) | Commodity: {row['Commodity']} | "
                f"Railcars: {row['Railcars-Containers']} | Rail-Miles: {row['Rail-Miles']} mi | "
                f"Net-Weight: {row['Net-Weight-Tons']} tons | Metric-Tonnes: {row['Metric-Tonnes']} | "
                f"Rail-km: {row['Rail-km']} km"
            ),
            "target": (
                f"Rail-GHG-Emissions-Tonnes: {row['Rail-GHG-Emissions-Tonnes']} | "
                f"Truck-GHG-Emissions-Tonnes: {row['Truck-GHG-Emissions-Tonnes']}"
            ),
        }
        for _, row in batch.iterrows()
    ]
	
    new_input = (
        f"Date: {test_case['Waybill Date']} | Origin: {test_case['Origin']} ({test_case['Origin-St']}) | "
        f"Destination: {test_case['Destination']} ({test_case['Destination-St']}) | Commodity: {test_case['Commodity']} | "
        f"Railcars: {test_case['Railcars-Containers']} | Rail-Miles: {test_case['Rail-Miles']} mi | "
        f"Net-Weight: {test_case['Net-Weight-Tons']} tons | Metric-Tonnes: {test_case['Metric-Tonnes']} | "
        f"Rail-km: {test_case['Rail-km']} km"
    )
    ground_truth = {
        "Rail": test_case["Rail-GHG-Emissions-Tonnes"],
        "Truck": test_case["Truck-GHG-Emissions-Tonnes"],
    }
	
    messages = [
        {"role": "system", "content": "You are an expert at predicting GHG emissions from logistics data. Provide precise predictions in the format: Rail-GHG-Emissions-Tonnes: <value> | Truck-GHG-Emissions-Tonnes: <value>"},
        {"role": "user", "content":
            "Predict GHG emissions for the following logistics scenario. Here are some example inputs and their corresponding emissions:\n\n" +
            "\n".join([f"Example: {ex['feature']}\nEmissions: {ex['target']}" for ex in k_shot_examples]) +
            f"\n\nPredict emissions for this input:\n{new_input}"
        }
    ]
	
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
	
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
	
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted_rail, predicted_truck = extract_emissions(output_text)
	
    result = {
        "Batch": batch_num + 1,
        "Ground Truth Rail": ground_truth["Rail"],
        "Ground Truth Truck": ground_truth["Truck"],
        "Predicted Rail": predicted_rail,
        "Predicted Truck": predicted_truck,
        "Generated Text": output_text,
    }
    results.append(result)
	
    print(f"Batch {batch_num + 1}:")
    print(f"Ground Truth - Rail: {ground_truth['Rail']} tonnes CO2e, Truck: {ground_truth['Truck']} tonnes CO2e")
    print(f"Prediction - Rail: {predicted_rail} tonnes CO2e, Truck: {predicted_truck} tonnes CO2e")
    print(f"Generated Text: {output_text}\n")

results_df = pd.DataFrame(results)
output_results_path = "prediction_results_llama.csv"
results_df.to_csv(output_results_path, index=False)

print(f"Prediction results saved at: {output_results_path}")