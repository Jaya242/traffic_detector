import gradio as gr
from src.detector import process_video

def run(video_input):
    if video_input is None:
        return None, None, "⚠️  Please upload a video first.", None, None
    return process_video(video_input)


demo = gr.Interface(                  
    fn=run,
    inputs=gr.Video(label="Upload a traffic video"),
    outputs=[
        gr.Video(label="Annotated Output (with tracking + counting lines)"),
        gr.File(label="Crossings CSV"),
        gr.Markdown(label="Summary"),
        gr.Plot(label="📊 Crossings by Class"),
        gr.Plot(label="⏱️  Crossings Over Time"),
    ],
    title="🚦 Traffic Analytics Pipeline",
    description=(
          "Upload a traffic intersection video. Get back annotated output with "
          "persistent vehicle tracking, dual-line crossing counter, and a CSV log "
          "of every crossing event. Built with YOLOv8 + ByteTrack + OpenCV."
    ),
    article=(
          "Code on [GitHub](https://github.com/Jaya242/traffic_detector). "
          "Validated at 96.2% unique-vehicle accuracy on a 2,208-frame test clip."
    ),
    flagging_mode="never",
)
  

if __name__ == "__main__":
    demo.launch()