import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("daiyizheng/valid")
launch_gradio_widget(module)