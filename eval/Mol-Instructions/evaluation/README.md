# Evaluation

## 🔬 Molecule

🚀 **Step 1:** Start by generating results based on the test set, similar to the `generate_example.py` function. Make sure to save these results in `output_dir`.

(✨ Please be aware: the `generate_example.py` code is specifically designed for the model tuned from [Llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat), with LoRA weights available at [llama2-molinst-molecule-7b](https://huggingface.co/zjunlp/llama2-molinst-molecule-7b).)

🔨 **Step 2:** Next, run the `evaluate.py` script to perform the preprocessing steps. Here, the `input_path` parameter should be set to `output_dir` from Step 1, `output_path` is where you wish to store the preprocessed data, and `task` denotes the specific molecular task you're running:
- For `task == 'property_pred'`, you'll execute molecular property prediction tasks.
- For `task == 'mol_gen'`, you'll undertake molecule generation tasks (including `description_guided_molecule_design`, `forward_reaction_prediction`, `retrosynthesis`, `reagent_prediction`).
- For `task == 'understand'`, you'll handle molecule understanding tasks, specifically `molecular_description_generation`.

📊 **Step 3:** For molecule generation tasks, continue by running:
```bash
python mol_translation_selfies.py --input_file $PREPROCESSED_DATA_FROM_STEP2$
python fingerprint_metrics.py --input_file $PREPROCESSED_DATA_FROM_STEP2$
```
And for molecule understanding tasks, execute:
```bash
python text_translation_metrics.py --input_file $PREPROCESSED_DATA_FROM_STEP2$
```



## 🧬 Protein

🚀 **Step 1:** Run the `protein/generate.py` and set `{task}` to perform the generation of corresponding task, and the result will be saved to `{task}_result.json`.

🔨 **Step 2:** Execute `protein/eval.py` to evaluate the performance of generations.


## 🥼 Biotext
🚀 **Step 1:** Start by generating results based on the test set, similar to the `generate_example.py` function. Make sure to save these results in `output_dir`.

🔨 **Step 2:** Based on the metrics you need to evaluate, select the corresponding file in the `biotext` folder and modify the path of `output_dir`.

📊 **Step 3:** Run the modified Python file you need.







## 🌟 Acknowledgements

Special thanks to [MolT5](https://github.com/blender-nlp/MolT5) for providing the evaluation code that greatly assisted our project.
