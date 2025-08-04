# 🧩 Maze Solver with Multiple Algorithms

## 1. Project Title

**Maze Solver with Flexible Maze Input and Algorithm Comparison**

This project allows users to solve mazes using classical pathfinding algorithms such as BFS, Dijkstra’s, and A*. It supports both pre-defined mazes and custom maze images.

---

## 2. Project Description

This Maze Solver application is an interactive tool that helps visualize pathfinding algorithms in action. Whether you're learning about graphs, algorithms, or working on AI, this tool provides hands-on experience.

### What it does:
- Solves mazes using BFS, Dijkstra, or A* algorithms.
- Supports predefined mazes and image-based mazes.
- Shows step-by-step pathfinding visually.
- Compares performance of each algorithm.

### Why these technologies:
- **Python**: Easy to implement and visualize algorithms.
- **Pygame**: Interactive GUI and visual effects.
- **NumPy**, **Matplotlib**, **scikit-image**: For efficient image and numerical processing.

### Challenges & Future Plans:
- Handling complex image mazes with noise.
- Improving performance on high-resolution inputs.
- Future enhancements like 3D mazes, exporting solutions, and mobile/web integration.

---

## 4. Technologies Used

- Python 3.x
- NumPy - Numerical computations and maze representation
- Pygame - Interactive GUI and visualization
- Matplotlib - Image handling and plotting
- scikit-image - Image processing and skeletonization
- Collections & Heapq - Data structures for algorithms

---

## 5. Prerequisites

Make sure you have Python 3.x installed on your system. You can download it from [https://www.python.org](https://www.python.org).

---

## 6. Installation

### Clone the repository:

```bash
git clone https://github.com/yourusername/maze-solver.git
cd maze-solver
```

### Install required dependencies:
```bash
pip install numpy pygame matplotlib scikit-image
```

---

## 7. Usage

Running the Application
```bash
python maze_solver.py
```

### Option 1: Pre-defined Maze

1. Select option **1** when prompted  
2. Choose maze size:
   - `9x9`
   - `16x16`
   - `21x21`
3. Click on the maze to set the **START** point (appears green)  
4. Click on another location to set the **END** point (appears red)  
5. Choose an algorithm:
   - `1` - Breadth-First Search (BFS)  
   - `2` - Dijkstra's Algorithm  
   - `3` - A* Search  
6. Watch the algorithm solve the maze in real-time  
7. Press `R` to reset and generate a new maze  

### Option 2: Image Maze

1. Select option **2** when prompted  
2. Enter the **relative path** to your maze image file  
3. Click on the displayed image to select **start** and **end** points  
4. Choose your preferred solving algorithm  
5. View the **solution overlaid** on your original maze image  


---

### 8. Algorithm Comparison

| Algorithm | Time Complexity     | Space Complexity | Optimal  | Description                                                        |
|-----------|---------------------|------------------|----------|--------------------------------------------------------------------|
| BFS       | O(V + E)            | O(V)             | ✅ Yes   | Explores all nodes at current depth before going deeper           |
| Dijkstra  | O((V + E) log V)    | O(V)             | ✅ Yes   | Finds shortest path considering edge weights                       |
| A*        | O(b^d)              | O(b^d)           | ✅ Yes*  | Uses heuristics to guide search (*if heuristic is admissible)     |

---

### 9. Supported Image Formats

The application supports the following common image formats:

- **PNG**
- **JPG / JPEG**
- **BMP**
- **TIFF**

#### Image Requirements:

- Black walls, white paths  
- Clear contrast between walls and pathways  
- Reasonable resolution for processing  

---

### 10. Performance Notes

- **Small mazes (9x9):** All algorithms perform similarly  
- **Medium mazes (16x16):** A* typically outperforms others  
- **Large mazes (21x21+):** A* shows significant advantage  
- **Image mazes:** Performance depends on image complexity and resolution

- ### 11. Customization

#### Color Scheme

You can modify the color scheme by editing the `colors` dictionary in the `draw_maze()` function:

```python
colors = {
    "wall": (0, 0, 0),             # Black walls
    "path": (255, 255, 255),       # White paths
    "start": (0, 255, 0),          # Green start
    "end": (255, 0, 0),            # Red end
    "explored": (173, 216, 230),   # Light blue explored
    "current": (255, 165, 0),      # Orange current
    "solution": (0, 0, 255),       # Blue solution path
}
```

### Algorithm Speed
Adjust visualization speed by modifying time.sleep() values inside the algorithm functions.

---

### 12. Troubleshooting

| Issue                                | Solution                                              |
|--------------------------------------|-------------------------------------------------------|
| `No module named 'pygame'`           | Run: `pip install pygame`                             |
| Could not find valid start or end points | Ensure you're clicking on white/path areas          |
| Image not loading                    | Check if the path is correct and image format is supported |
| Slow performance on large images     | Resize images or use smaller maze sizes              |

---

### 13. Future Enhancements

- Additional algorithms (e.g., Greedy Best-First Search, Jump Point Search)  
- 3D maze support  
- Maze difficulty rating system  
- Export solution as image/video  
- Web-based interface  
- Mobile app version  










