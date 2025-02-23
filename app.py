import os
import random
import asyncio
import platform
import gradio as gr
from model import teeniefier

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.nn.functional import softmax

class TeeniePingApp:
    def __init__(self):
        self.image_size = 500
        self.ping_image_folder = 'dataset/test'
        self.total_stage = 5
        self.start_stage = 0
        self.start_score = 0
        self.start_ai_score = 0
        self.target_ping = ''
        self.count_down_start = 3 
        
        system = platform.system()

        if system == 'Darwin':  # macOS
            self.font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        elif system == 'Windows':  # Windows
            self.font_path = 'C:\\Windows\\Fonts\\Arial.ttf'
        else:  # Linux 
            font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
            if not os.path.exists(font_path):
                import subprocess   
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y', 'fonts-noto-cjk'], check=True)
            self.font_path = font_path

        self.model = self.load_model()

        self.demo = gr.Blocks()
        self.setup_ui()

    def load_model(self):
        model = teeniefier(num_teenieping=len(os.listdir(self.ping_image_folder)))

        trained_model_path = 'best_model.ckpt'
        trained_model = torch.load(trained_model_path)
        model.load_state_dict(trained_model['model'])
        model.idx2ping = trained_model['classes']
        return model

    def preprocess_image(self, image_path):
        image_origin = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.6151984, 0.51760532, 0.46836003), 
                                (0.26411435, 0.24187316, 0.264022790))
        ])
        image = transform(image_origin)
        return image.unsqueeze(0)

    async def infer_image(self, image_path):
        image = self.preprocess_image(image_path)
        output = self.model(image)
        prob = softmax(output, dim=1)
        value, index = torch.max(prob, dim=1)
        return self.model.idx2ping[index.item()], value.item()

    def create_countdown_image(self, text):
        img = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(self.font_path, size=100)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.image_size - text_width) // 2
        y = (self.image_size - text_height) // 2

        draw.text((x, y), text, font=font, fill="black")
        return img

    def sample_ping(self):
        target_ping = random.sample(os.listdir(self.ping_image_folder), 1)[0]
        target_image = random.sample(os.listdir(os.path.join(self.ping_image_folder, target_ping)), 1)[0]

        return target_ping, target_image

    def resize_ping_image(self, ping_image):
        ping_width, ping_height = ping_image.size
        max_length = max(ping_width, ping_height)
        ratio = self.image_size / max_length
        ping_image = ping_image.resize((int(ping_width * ratio), int(ping_height * ratio)))
        return ping_image

    def create_text_image(self, text):
        img = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(self.font_path, size=30)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.image_size - text_width) // 2
        y = (self.image_size - text_height) // 2

        draw.text((x, y), text, font=font, fill="black")
        return img

    def draw_left_box(self, current_stage, content_image, current_score=0):
        background_pil = Image.new("RGB", 
                                   (self.image_size, self.image_size), 
                                   color=(255, 255, 255))

        if current_stage > self.total_stage:
            text = f'''당신의 점수는 : \n {current_score} 점!!'''
            background_pil = self.create_text_image(text)
        
        elif current_stage > 0 and content_image is not None:
            central_x = self.image_size // 2
            central_y = self.image_size // 2

            background_pil.paste(content_image, 
                                 (central_x - content_image.size[0] // 2, 
                                  central_y - content_image.size[1] // 2))

        background = gr.Image(background_pil)
        return background

    def write_right_button(self, stage_count):
        if stage_count > self.total_stage:
            return '끝!!'
        else:
            return f'다음으로! ({stage_count}/{self.total_stage})'

    async def right_button_clicked(self, current_stage, ai_score, current_score):
        next_stage = current_stage + 1

        if next_stage > self.total_stage:
            final_score_text = f"AI의 점수는: \n{ai_score}점"
            final_image = self.create_text_image(final_score_text)
            yield (
                self.draw_left_box(next_stage, None, current_score),
                gr.update(value='끝!!', interactive=False),
                gr.update(interactive=False),
                gr.update(visible=True),
                gr.Image(final_image),
                next_stage,
                ai_score
            )
        else:
            self.target_ping, target_image = self.sample_ping()

            ping_image = Image.open(os.path.join(self.ping_image_folder, 
                                                 self.target_ping, 
                                                 target_image)).convert('RGB')
            ping_image = self.resize_ping_image(ping_image)
            left_box_image = self.draw_left_box(next_stage, ping_image)

            countdown_numbers = list(range(self.count_down_start, 0, -1))
            for number in countdown_numbers:
                countdown_img = self.create_countdown_image(str(number))
                yield (
                    left_box_image,
                    gr.update(value=self.write_right_button(next_stage)),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                    gr.Image(countdown_img),
                    next_stage,
                    ai_score
                )
                await asyncio.sleep(1)

            image_path = os.path.join(self.ping_image_folder, self.target_ping, target_image)
            ai_result, confidence = await self.infer_image(image_path)
            infer_and_ping_name_text = f"이 티니핑의 이름은 \n '{self.target_ping}' 입니다! \n\n AI의 예측은?? '{ai_result}'!"
            if ai_result == self.target_ping: 
                ai_score += 1
            right_box_image = self.create_text_image(infer_and_ping_name_text)

            yield (
                left_box_image,
                gr.update(value=self.write_right_button(next_stage)),
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.Image(right_box_image),
                next_stage,
                ai_score
            )
        

    def left_button_clicked(self, current_score):
        current_score += 1
        return gr.update(interactive=False), current_score

    def play_again_clicked(self):
        current_stage = 0
        current_score = 0
        ai_score = 0 
        
        left_box_image = self.draw_left_box(current_stage, None)
        
        return (
            left_box_image,
            gr.Button(value='시작하기', interactive=True),
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.Image(Image.new("RGB", 
                               (self.image_size, self.image_size), 
                               color=(255, 255, 255))),
            current_stage,
            current_score,
            ai_score
        )
    
    def test_text_clicked(self, test_state):
        test_state += 1
        return gr.update(value=test_state), test_state
    
    def setup_ui(self):
        with self.demo:
            with gr.Row():
                Title = gr.HTML("<div style='text-align: center;'><h1>티니핑 대결!!</h1></div>")
            with gr.Row():
                with gr.Column():
                    self.left_box = self.draw_left_box(0, None)
                with gr.Column():
                    self.right_box = gr.Image(Image.new("RGB", size=(self.image_size, self.image_size),
                                                       color=(255, 255, 255)))
            with gr.Row():
                with gr.Column():
                    self.left_button = gr.Button(value='맞았다!👍', interactive=False, visible=True)
                with gr.Column():
                    self.right_button = gr.Button(value='시작하기', interactive=True)
            with gr.Row():
                self.play_again_button = gr.Button(value='한판 더!🚀', visible=False, interactive=True)

            self.current_stage = gr.State(value=self.start_stage)
            self.current_score = gr.State(value=self.start_score)
            self.ai_score = gr.State(value=self.start_ai_score)

            self.left_button.click(
                self.left_button_clicked,
                inputs=[self.current_score],
                outputs=[self.left_button, 
                         self.current_score]
            )
            self.right_button.click(
                self.right_button_clicked,
                inputs=[self.current_stage, self.ai_score, self.current_score],
                outputs=[self.left_box, 
                         self.right_button, 
                         self.left_button, 
                         self.play_again_button, 
                         self.right_box, 
                         self.current_stage,
                         self.ai_score]
            )
            self.play_again_button.click(
                self.play_again_clicked,
                outputs=[self.left_box, 
                         self.right_button, 
                         self.left_button, 
                         self.play_again_button,
                         self.right_box,
                         self.current_stage,
                         self.current_score,
                         self.ai_score]
            )

    def launch(self):
        self.demo.launch()

if __name__ == "__main__":
    app = TeeniePingApp()
    app.launch()