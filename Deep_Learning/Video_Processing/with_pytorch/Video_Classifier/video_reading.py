import os
import cv2
"""pip install opencv-python"""
# Dataset folder
dataset_path = "dataset"

# Read all action folders
classes = sorted(os.listdir(dataset_path))

print("=" * 60)
print("Classes Found")
print("=" * 60)

for class_name in classes:

    print(f"\nClass : {class_name}")

    class_folder = os.path.join(dataset_path, class_name)

    videos = os.listdir(class_folder)

    for video in videos:

        if not video.endswith(".mp4"):
            continue

        video_path = os.path.join(class_folder, video)

        print(f"\nReading Video : {video}")
        print(f"Full Path     : {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Cannot open video")
            continue

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            print(
                f"Frame {frame_count} "
                f"Shape = {frame.shape}"
            )

            cv2.imshow(class_name, frame)

            # Press q to skip to next video
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"Total Frames : {frame_count}")

print("\nFinished Reading All Videos")