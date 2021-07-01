import gradio as gr

description = "demo for ByT5-base fine-tuned for Question Answering (on Tweets). To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2105.13626'>ByT5: Towards a token-free future with pre-trained byte-to-byte models</a> | <a href='https://huggingface.co/Narrativa/byt5-base-finetuned-tweet-qa'>Huggingface Model</a></p> | <a href='https://github.com/google-research/byt5/'>Github Repo</a></p> "



gr.Interface.load("huggingface/Narrativa/byt5-base-finetuned-tweet-qa", inputs=gr.inputs.Textbox(lines=5)).launch()

