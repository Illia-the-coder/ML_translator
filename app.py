import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes
import time
import docx2txt

model_name_dict = {
        'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M',
                  #'nllb-1.3B': 'facebook/nllb-200-1.3B',
                  # 'nllb-distilled-1.3B': 'facebook/nllb-200-distilled-1.3B',
                  #'nllb-3.3B': 'facebook/nllb-200-3.3B',
                  }

def load_models():
    # build model and tokenizer
    model_dict = {}

    device =  'cuda' if torch.cuda.is_available() else 'cpu'

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name, torch_dtype=torch.float32).to(device)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer

    return model_dict

def read_file(file):
    # Read content from a file
    if file.name.endswith('.txt'):
        with open(file.name, 'r', encoding='utf-8') as f:
            return f.read()
    elif file.name.endswith('.docx'):
        return docx2txt.process(file.name)
    else:
        return None


def translation(model_name, source, target, file=None, progress=gr.Progress()):
    if len(model_dict) == 0:
        return {"error": "Models not loaded"}

    language_map = {
        "English (US)": "eng_Latn",
        "Russian": "rus_Cyrl",
        "Spanish": "spa_Latn",
        "French": "fra_Latn",
        "Portuguese (Brazil)": "por_Latn"
    }

    start_time = time.time()
    source = language_map[source]
    target = language_map[target]

    if file is not None:
        text = read_file(file)
        if text is None:
            return {"error": "Unsupported file format"}


    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target)

    result = ''
    n = 1
    for x in text.split('\n\n'):
      if x.strip() != '':
        for text_chunk in x.split('.'):
            print(text_chunk)
            translation = translator(text_chunk, max_length=512)[0]['translation_text']
            result += translation   
        result += '\n\n'
        n += 1
        progress(n / len(text.split('\n\n')))

    print(result)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    return 'output.txt'


if __name__ == '__main__':

    global model_dict

    model_dict = load_models()

   # Define Gradio demo
    model_names = model_name_dict.keys()

    source_languages = [
        "English (US)",
        "Russian",
        "Spanish"
    ]

    target_languages = [
        "English (US)",
        "Spanish",
        "French",
        "Russian",
        "Portuguese (Brazil)"
    ]

    inputs = [
        gr.Dropdown(model_names, label='NLLB Model', value='nllb-distilled-600M'),
        gr.Dropdown(source_languages, label='Source Language', value="English (US)"),
        gr.Dropdown(target_languages, label='Target Language'),
        gr.File(label="Upload file (txt or docx format)", file_types=['.txt', '.docx']),
    ]

    outputs = gr.File()

    title = "NLLB Multi-Model Translation"
    description = "Translate text using various NLLB models. Details: https://github.com/facebookresearch/fairseq/tree/nllb."

    gr.Interface(
        fn=translation,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description
    ).launch(debug=True, share=True)
