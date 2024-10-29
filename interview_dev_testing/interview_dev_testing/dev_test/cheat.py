import cv2
import mediapipe as mp
from pydub import AudioSegment
from pydub.playback import play

def cam():
    # Initialize MediaPipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    sound = AudioSegment.from_mp3(r"E:\code_files\Banao_Intern\AI_Interview\alert_gas_leak.mp3")

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    warning_msg_shown = False  # Variable to track if warning message is shown

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,  # Adjust max_num_faces to 2 to detect multiple faces
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

            if num_faces > 1:
                if not warning_msg_shown:
                    print("2 faces detected...")  # Show warning message
                    play(sound)
                    warning_msg_shown = True
            elif num_faces == 0:
                if not warning_msg_shown:
                    print("Please show your face...")  # Show warning message
                    play(sound)
                    warning_msg_shown = True
            else:
                if warning_msg_shown:
                    warning_msg_shown = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face landmarks
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    )
                    # Get specific landmarks for left and right eyes
                    left_eye_landmarks  = face_landmarks.landmark[468:473]
                    right_eye_landmarks = face_landmarks.landmark[473:478]
                    # Draw circles on eye landmarks
                    for landmark in left_eye_landmarks + right_eye_landmarks:
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Display the image
            resized_frame = cv2.resize(image, (800, 600)) 
            cv2.imshow("Face Mesh", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam()