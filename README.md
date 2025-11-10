
📐 Shape Detector: Low-Level Vision Pipeline
This project is a Shape Detection and Classification system built entirely in TypeScript/JavaScript, implementing classical computer vision algorithms from scratch (no external libraries like OpenCV).

Core Functionality
The pipeline processes raw image data to identify and classify simple geometric forms:

*Blob Detection: Uses Flood Fill (BFS) to segment objects from the background.

*Feature Extraction: Generates the Convex Hull and various metrics (Area, Solidity, Perimeter) for each object.

*Classification: Employs the Ramer-Douglas-Peucker (RDP) algorithm to accurately count corners, distinguishing between Triangles, Rectangles, Pentagons, and Stars.

Technical Focus
The main challenge was accurately tuning the parameters (the RDP epsilon and classification thresholds) to handle noise and pixelation, ensuring high-fidelity detection for both sharp-cornered polygons and smooth circles.
