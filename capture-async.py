import os
import cv2
import time
import base64
import numpy as np
import asyncio
import aiohttp
import threading
import queue
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")


def encode_image(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


async def guess_clue(session, content) -> None:
    try:
        print("🤖 Time to guess!")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": """
                    You are a charades expert.
                    I will send you frames of a video recording of a person, frame by frame, and you will
                    try to guess what they are acting out. Please only your guess in your response and nothing else. 
                    """,
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": 300,
        }

        response = await session.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response = await response.json()
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"An error occurred: {e}")

def api_call_thread(frame_queue, stop_event):
    asyncio.run(api_call_coroutine(frame_queue, stop_event))

async def api_call_coroutine(frame_queue, stop_event):
    content = [{"type": "text", "text": "What am I acting out?"}]
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            if not frame_queue.empty():
                base64_image = frame_queue.get()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
                # await guess_clue(session, content)
            await asyncio.sleep(3)  # Check for new frame every 3 seconds

def main():
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    api_thread = threading.Thread(target=api_call_thread, args=(frame_queue,stop_event,))
    api_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Video Feed", frame)

            if time.time() - last_capture_time > 3:
                last_capture_time = time.time()

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                max_size = 250
                ratio = max_size / max(pil_img.size)
                new_size = tuple([int(x*ratio) for x in pil_img.size])
                resized_img = pil_img.resize(new_size, Image.LANCZOS)
                frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
                base64_image = encode_image(frame)

                frame_queue.put(base64_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()
    api_thread.join()

if __name__ == "__main__":
    main()
