import os
import pickle
import numpy as np
import rclpy

from rclpy.node import Node

from std_msgs.msg import String, Int16

from ament_index_python.packages import (
    get_package_share_directory
)

from .recorder import recorder

from .voice_processing import (
    FRAME_SIZE,
    HOP_SIZE,
    ORDER,
    N_STATES,
    FREQ,
    preprocess_audio,
    extract_mfcc,
    recognize_command
)

# =========================
# ROS2 NODE
# =========================

class VoiceRecognitionNode(Node):

    def __init__(self):

        super().__init__(
            "voice_recognition_node"
        )

        # =========================
        # Publishers
        # =========================

        self.command_publisher = (
            self.create_publisher(
                String,
                "voice_command",
                10
            )
        )

        self.index_publisher = (
            self.create_publisher(
                Int16,
                "command_index",
                10
            )
        )

        # =========================
        # Load models
        # =========================

        package_path = (
            get_package_share_directory(
                "your_package_name"
            )
        )

        codebook_path = os.path.join(
            package_path,
            "models",
            "codebook.pkl"
        )

        models_path = os.path.join(
            package_path,
            "models",
            "models.pkl"
        )

        with open(codebook_path, "rb") as f:

            self.kmeans = pickle.load(f)

        with open(models_path, "rb") as f:

            self.models = pickle.load(f)

        # =========================
        # Command mapping
        # =========================

        self.command_to_index = {

            "start": 0,
            "pause": 1,
            "next": 2,
            "stop": 3
        }

        # =========================
        # Timer
        # =========================

        self.timer = self.create_timer(
            2.0,
            self.recognition_callback
        )

        self.get_logger().info(
            "Voice recognition node started."
        )

    # =========================
    # Recognition Callback
    # =========================

    def recognition_callback(self):

        try:

            self.get_logger().info(
                "Recording audio..."
            )

            # =========================
            # Record audio
            # =========================

            audio = recorder()

            # =========================
            # Preprocess
            # =========================

            audio = preprocess_audio(audio)

            if len(audio) == 0:

                self.get_logger().warning(
                    "Empty audio"
                )

                return

            # =========================
            # MFCC
            # =========================

            mfcc = extract_mfcc(
                audio,
                FREQ
            )

            if len(mfcc) < N_STATES:

                self.get_logger().warning(
                    "Too few frames"
                )

                return

            # =========================
            # Vector Quantization
            # =========================

            observations = (
                self.kmeans.predict(mfcc)
            )

            # =========================
            # Recognition
            # =========================

            prediction, score, scores = (
                recognize_command(
                    observations,
                    self.models
                )
            )

            # =========================
            # Publish command
            # =========================

            command_msg = String()

            command_msg.data = prediction

            self.command_publisher.publish(
                command_msg
            )

            # =========================
            # Publish index
            # =========================

            index_msg = Int16()

            index_msg.data = (
                self.command_to_index[
                    prediction
                ]
            )

            self.index_publisher.publish(
                index_msg
            )

            # =========================
            # Log
            # =========================

            self.get_logger().info(
                f"Prediction: {prediction}"
            )

            self.get_logger().info(
                f"Score: {score}"
            )

        except Exception as e:

            self.get_logger().error(
                str(e)
            )

# =========================
# MAIN
# =========================

def main(args=None):

    rclpy.init(args=args)

    node = VoiceRecognitionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":

    main()