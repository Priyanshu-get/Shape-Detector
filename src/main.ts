import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}


export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM IMPLEMENTATION
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    // 1. Convert image to a 2D binary grid (0 = object, 255 = background)
    const binaryImage = this.binarize(imageData, 128);

    // 2. Find all connected "blobs" (groups of object pixels)
    const blobs = this.findBlobs(binaryImage);

    const shapes: DetectedShape[] = [];
    const { width, height } = imageData;

    // 3. Classify each blob
    for (const blob of blobs) {
      // 4. Get all properties and classify
      const shape = this.classifyBlob(blob, width, height);
      if (shape) {
        shapes.push(shape);
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  /**
   * Converts RGBA ImageData to a 2D binary grid.
   * Respects transparency.
   * @returns 2D array where 0 = object, 255 = background
   */
  private binarize(imageData: ImageData, threshold: number): number[][] {
    const { width, height, data } = imageData;
    const grid: number[][] = Array.from({ length: height }, () =>
      Array(width).fill(255)
    );

    for (let i = 0; i < data.length; i += 4) {
      const x = (i / 4) % width;
      const y = Math.floor(i / 4 / width);

      const alpha = data[i + 3];

      // Treat transparent or semi-transparent pixels as background
      if (alpha < 128) {
        grid[y][x] = 255; // Background
        continue;
      }

      // Calculate grayscale value
      const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;

      // Apply threshold
      grid[y][x] = gray < threshold ? 0 : 255; // 0 = Object, 255 = Background
    }
    return grid;
  }

  
   //* Finds all connected-component blobs in a binary image.
  
  private findBlobs(binaryImage: number[][]): Point[][] {
    const height = binaryImage.length;
    const width = binaryImage[0].length;
    const labels: number[][] = Array.from({ length: height }, () =>
      Array(width).fill(0)
    );
    const blobs: Point[][] = [];
    let currentLabel = 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // If we find an object pixel (0) that hasn't been labeled yet
        if (binaryImage[y][x] === 0 && labels[y][x] === 0) {
          // Start a new blob
          const blob = this.floodFill(
            binaryImage,
            x,
            y,
            labels,
            currentLabel
          );
          blobs.push(blob);
          currentLabel++;
        }
      }
    }
    return blobs;
  }


  // * Performs a flood-fill (BFS) to find all pixels in a single blob.
  
  private floodFill(
    binaryImage: number[][],
    startX: number,
    startY: number,
    labels: number[][],
    label: number
  ): Point[] {
    const height = binaryImage.length;
    const width = binaryImage[0].length;
    const queue: Point[] = [{ x: startX, y: startY }];
    const blob: Point[] = [];
    labels[startY][startX] = label;

    // 8-way connectivity (neighbors)
    const deltas = [
      [0, 1], [0, -1], [1, 0], [-1, 0],
      [1, 1], [1, -1], [-1, 1], [-1, -1],
    ];

    while (queue.length > 0) {
      const point = queue.shift()!;
      blob.push(point);

      for (const [dx, dy] of deltas) {
        const nx = point.x + dx;
        const ny = point.y + dy;

        // Check bounds and if it's an unlabeled object pixel
        if (
          nx >= 0 && nx < width && ny >= 0 && ny < height &&
          binaryImage[ny][nx] === 0 && labels[ny][nx] === 0
        ) {
          labels[ny][nx] = label;
          queue.push({ x: nx, y: ny });
        }
      }
    }
    return blob;
  }

 
  // * Main classification logic. Analyzes a blob to determine its shape.
  
  private classifyBlob(blob: Point[], imgWidth: number, imgHeight: number): DetectedShape | null {
    // 1. Get basic properties
    const { area, center, boundingBox } = this.getBlobProperties(blob);

    // Filter out noise
    if (area < 50 || boundingBox.width < 10 || boundingBox.height < 10) {
      return null;
    }

    // 2. Find the boundary pixels
    const boundary = this.findBoundary(blob, imgWidth, imgHeight);
    if (boundary.length < 10) return null; 

    // 3. Find the Convex Hull of the boundary
    const hull = this.convexHull(boundary);
    if (hull.length < 3) return null;

    // 4. Calculate hull features
    const hullArea = this.polygonArea(hull);
    const hullPerimeter = this.polygonPerimeter(hull);
    
    if (hullArea === 0 || hullPerimeter === 0) return null;

    // 5. CLASSIFICATION LOGIC
    
    // 5a. Check Solidity: area of pixels / area of convex hull
    // This is the best check for concavity (stars).
    const solidity = area / hullArea;
    if (solidity < 0.85) { 
      return {
        type: "star",
        confidence: Math.max(0.5, (0.85 - solidity) / 0.4),
        boundingBox,
        center,
        area,
      };
    }

    // 5b. It's a convex shape. Check for polygon corners FIRST.
    // is a good starting point (more sensitive to corners).
    const epsilon = hullPerimeter * 0.025; 
    
    // Add the first point to the end to "close" it.
    const closedHull = [...hull, hull[0]];
    const corners = this.approxPolyDP(closedHull, epsilon);
    
    // The result will include the duplicated closing point,
    // so we subtract 1.
    const cornerCount = corners.length - 1;

    let type: DetectedShape["type"] | null = null;
    let confidence = 0.9;

    switch (cornerCount) {
      case 3:
        type = "triangle";
        break;
      case 4:
        type = "rectangle";
        break;
      case 5:
        type = "pentagon";
        break;
      default:
        // It's not a simple polygon. NOW we check for circularity.
        // This prevents pentagons (cornerCount=5) from ever
        // reaching this check.
        const circularity = (hullPerimeter * hullPerimeter) / (4 * Math.PI * hullArea);
        
        // Use a forgiving circularity threshold.
        if (circularity < 1.25) { 
          type = "circle";
          confidence = Math.max(0.5, 1.0 - (circularity - 1.0) * 4.0);
        }
    }

    if (type) {
      return { type, confidence, boundingBox, center, area };
    }

    return null;
  }


   //* Calculates Area, Center (Centroid), and Bounding Box for a blob.

  private getBlobProperties(blob: Point[]) {
    const area = blob.length;
    let sumX = 0;
    let sumY = 0;
    let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;

    for (const p of blob) {
      sumX += p.x;
      sumY += p.y;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    const center = { x: sumX / area, y: sumY / area };
    const boundingBox = {
      x: minX,
      y: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
    };

    return { area, center, boundingBox };
  }


   //* Finds pixels in a blob that are adjacent to the background.
  
  private findBoundary(blob: Point[], imgWidth: number, imgHeight: number): Point[] {
    const boundary: Point[] = [];
    const blobSet = new Set(blob.map(p => `${p.x},${p.y}`));
    const deltas = [[0, 1], [0, -1], [1, 0], [-1, 0]]; // 4-way

    for (const p of blob) {
      let isBoundary = false;
      for (const [dx, dy] of deltas) {
        const nx = p.x + dx;
        const ny = p.y + dy;
        // Check if neighbor is out of bounds (is boundary)
        // or if neighbor is not in the blob (is background)
        if (
          nx < 0 || nx >= imgWidth || ny < 0 || ny >= imgHeight ||
          !blobSet.has(`${nx},${ny}`)
        ) {
          isBoundary = true;
          break;
        }
      }
      if (isBoundary) {
        boundary.push(p);
      }
    }
    return boundary;
  }

  // --- Start of Geometry/Math Helper Functions ---

  
  // * Calculates the Convex Hull of a set of points using the Monotone Chain algorithm.
 
  private convexHull(points: Point[]): Point[] {
    if (points.length < 3) return points;

    // Sort points lexicographically (by x, then y)
    const sortedPoints = [...points].sort((a, b) =>
      a.x === b.x ? a.y - b.y : a.x - b.x
    );

    const upper: Point[] = [];
    const lower: Point[] = [];

    // Build upper hull
    for (const p of sortedPoints) {
      while (
        upper.length >= 2 &&
        this.crossProduct(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
      ) {
        upper.pop();
      }
      upper.push(p);
    }

    // Build lower hull
    for (let i = sortedPoints.length - 1; i >= 0; i--) {
      const p = sortedPoints[i];
      while (
        lower.length >= 2 &&
        this.crossProduct(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
      ) {
        lower.pop();
      }
      lower.push(p);
    }

    // Remove duplicates (first and last points of each chain)
    lower.pop();
    upper.pop();
    
    return upper.concat(lower);
  }

  /**
   * 2D cross product (for hull algorithm).
   * > 0: p1, p2, p3 make a "left" turn
   * < 0: "right" turn
   * = 0: collinear
   */
  private crossProduct(p1: Point, p2: Point, p3: Point): number {
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
  }

  
  // * Calculates polygon area using the Shoelace formula.
  
  private polygonArea(polygon: Point[]): number {
    let area = 0;
    let j = polygon.length - 1;
    for (let i = 0; i < polygon.length; i++) {
      area += (polygon[j].x + polygon[i].x) * (polygon[j].y - polygon[i].y);
      j = i;
    }
    return Math.abs(area / 2);
  }

  
   //* Calculates the perimeter of a polygon.
   
  private polygonPerimeter(polygon: Point[]): number {
    let perimeter = 0;
    let j = polygon.length - 1;
    for (let i = 0; i < polygon.length; i++) {
      perimeter += this.dist(polygon[i], polygon[j]);
      j = i;
    }
    return perimeter;
  }


  // * Euclidean distance between two points.
  
  private dist(p1: Point, p2: Point): number {
    return Math.sqrt(this.distSq(p1, p2));
  }


  // * Squared Euclidean distance (faster, avoids sqrt).
  
  private distSq(p1: Point, p2: Point): number {
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;
  }


   //* Implements the Ramer-Douglas-Peucker (approxPolyDP) algorithm.
   
  private approxPolyDP(points: Point[], epsilon: number): Point[] {
    if (points.length < 2) return points;
    const sqEpsilon = epsilon * epsilon;

    // Find the point with the maximum distance
    let dmax = 0;
    let index = 0;
    const end = points.length - 1;

    for (let i = 1; i < end; i++) {
      const d = this.getPerpendicularDistSq(points[i], points[0], points[end]);
      if (d > dmax) {
        index = i;
        dmax = d;
      }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (dmax > sqEpsilon) {
      const recResults1 = this.approxPolyDP(points.slice(0, index + 1), epsilon);
      const recResults2 = this.approxPolyDP(points.slice(index), epsilon);
      
      // Combine results, removing the duplicate middle point
      return recResults1.slice(0, recResults1.length - 1).concat(recResults2);
    } else {
      // All points are within epsilon, so return just the start and end points
      return [points[0], points[end]];
    }
  }

  /**
   * Helper for approxPolyDP. Finds squared perpendicular distance
   * from a point `p` to the line segment `p1` to `p2`.
   */
  private getPerpendicularDistSq(p: Point, p1: Point, p2: Point): number {
    let l2 = this.distSq(p1, p2);
    if (l2 === 0) return this.distSq(p, p1); // p1 and p2 are the same
    
    // Project p onto the line defined by p1 and p2
    let t = ((p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y)) / l2;
    t = Math.max(0, Math.min(1, t)); // Clamp to the line segment

    const projection = {
        x: p1.x + t * (p2.x - p1.x),
        y: p1.y + t * (p2.y - p1.y)
    };

    return this.distSq(p, projection);
  }

  // --- End of Geometry/Math Helper Functions ---


  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        // --- SVG HACK ---
        // SVGs might not have an intrinsic width/height.
        // We'll use a fixed size or the viewbox if possible.
        // For this challenge, let's assume a reasonable default
        // or that the SVG has width/height attributes.
        let { width, height } = img;
        
        // If it's an SVG and dimensions are 0, try to get from viewBox
        // or set a default.
        if (file.type === "image/svg+xml" && (!width || !height)) {
            // A simple default for test images
            width = width || 500;
            height = height || 500;
        }

        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx.clearRect(0, 0, width, height); // Clear previous image
        this.ctx.drawImage(img, 0, 0, width, height);
        const imageData = this.ctx.getImageData(0, 0, width, height);
        resolve(imageData);
      };
      img.onerror = (err) => {
          console.error("Image loading error:", err);
          reject(new Error("Could not load image"));
      };
      img.src = URL.createObjectURL(file);
    });
  }
}




class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});