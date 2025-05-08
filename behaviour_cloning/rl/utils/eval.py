import torch
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import signal
import imageio
from datetime import datetime

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Evaluation timed out")

class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len
        self.writer = None
        self.viewer = None

        if args.out_dir is None:
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)
        try:
            Path.mkdir(video_outdir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            print("Could not create video writer:", e)
            exit(-1)

    @torch.no_grad()
    def run(self):
        # Set up timeout to prevent hanging
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []

        # Make a camera.
        cam = self.viewer.cam
        mujoco.mjv_defaultCamera(cam)
        cam.elevation = -20
        cam.distance = 4

        reset_counter = 0
        try:
            observation = self.env.reset()
            while self.env.data.time < self.ep_len:
                step_start = time.time()

                # forward pass and step
                raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
                observation, reward, done, _ = self.env.step(raw.copy())

                # render scene
                cam.lookat = self.env.data.body(1).xpos.copy()
                renderer.update_scene(self.env.data, cam)
                pixels = renderer.render()
                frames.append(pixels)

                self.viewer.sync()

                if done and reset_counter < 3:
                    observation = self.env.reset()
                    reset_counter += 1

                time_until_next_step = max(
                    0, self.env.frame_skip*self.env.model.opt.timestep - (time.time() - step_start))
                time.sleep(time_until_next_step)

        except TimeoutException:
            print("Evaluation timed out - continuing with training")
        except Exception as e:
            print(f"Evaluation error: {e}")
        finally:
            # Cancel the timeout
            signal.alarm(0)
            
            # Always ensure resources are properly cleaned up
            if frames:
                try:
                    for frame in frames:
                        self.writer.append_data(frame)
                except Exception:
                    pass
                    
            if self.writer:
                try:
                    self.writer.close()
                except Exception:
                    pass
                    
            if self.viewer:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                    
            try:
                self.env.close()
            except Exception:
                pass
                
            print("Evaluation completed, resources cleaned up")
