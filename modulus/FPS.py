import time

class FPS:
    def __init__(self):
        self.counter = 0
        self.start_time = time.time()
    def frameСounter(self):
        self.counter += 1
        if (time.time() - self.start_time) > 1 :
            self.fps = self.counter / (time.time() - self.start_time)
            self.counter = 0
            self.start_time = time.time()
            return(self.fps)