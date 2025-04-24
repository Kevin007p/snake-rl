document.addEventListener("DOMContentLoaded", function () {
  // Full Vision Grid (10x10)
  const fullGridContainer = document.querySelector(".full-grid");
  const fullGridSize = 10;

  // Sample full grid state
  const fullGrid = Array(fullGridSize)
    .fill()
    .map(() => Array(fullGridSize).fill(0));

  // Add walls
  for (let i = 0; i < fullGridSize; i++) {
    fullGrid[0][i] = -1; // Top wall
    fullGrid[fullGridSize - 1][i] = -1; // Bottom wall
    fullGrid[i][0] = -1; // Left wall
    fullGrid[i][fullGridSize - 1] = -1; // Right wall
  }

  // Add snake (example position)
  fullGrid[5][5] = -0.5; // Snake head
  fullGrid[5][6] = -0.5; // Snake body
  fullGrid[5][7] = -0.5; // Snake body

  // Add food (example position)
  fullGrid[2][2] = 1.0;

  // Generate full grid cells
  for (let y = 0; y < fullGridSize; y++) {
    for (let x = 0; x < fullGridSize; x++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";

      const value = fullGrid[y][x];
      if (value === -1) {
        cell.classList.add("wall");
      } else if (value === -0.5) {
        cell.classList.add("snake");
      } else if (value === 1) {
        cell.classList.add("food");
      } else {
        cell.classList.add("empty");
      }

      fullGridContainer.appendChild(cell);
    }
  }

  // Limited Vision Grid (5x5)
  const limitedGridContainer = document.querySelector(".limited-grid");
  const limitedGridSize = 5;

  // Sample limited grid state (centered on snake head)
  const limitedGrid = [
    [-1, -1, -1, -1, -1],
    [-1, 0, 0, 0, -1],
    [-1, 0, -0.5, 0, -1],
    [-1, 0, 1, 0, -1],
    [-1, -1, -1, -1, -1],
  ];

  // Generate limited grid cells
  for (let y = 0; y < limitedGridSize; y++) {
    for (let x = 0; x < limitedGridSize; x++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";

      const value = limitedGrid[y][x];
      if (value === -1) {
        cell.classList.add("wall");
      } else if (value === -0.5) {
        cell.classList.add("snake");
      } else if (value === 1) {
        cell.classList.add("food");
      } else {
        cell.classList.add("empty");
      }

      cell.textContent = value;
      limitedGridContainer.appendChild(cell);
    }
  }
});
