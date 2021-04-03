import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

preTime = 0
def getFps():
    global preTime
    cTime = time.time()
    fps = 1/(cTime - preTime)
    preTime = cTime
    return fps

def main():
    while True:
        sucess, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w) , int(lm.y*h)
                    if id == 4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        fps = getFps()
        cv2.putText(img, str(int(fps)),(60, 60),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
