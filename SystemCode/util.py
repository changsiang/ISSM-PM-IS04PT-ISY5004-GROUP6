import cv2

class util:
    POSIITONS = {
        "top_left": (0, 0),
        "top_right": (0, 1),
        "bottom_left": (1, 0),
        "bottom_right": (1, 1)
    }
    def __init__(self) -> None:

        pass

    def draw_text(self, frame, position=None, color=(255, 0, 0), size=0.9, thickness=2, text="Hello World"):
        if position == None:
            position = self.POSIITONS["top_left"]
        x, y = position
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, size, color, thickness)
        return frame
    
    def draw_box(self, frame, position=None, color=(0, 255, 0)):
        x, y, w, h = position
        cv2.rectangle(frame, (x, y), (x + w, y + w), color, 2)
        return frame