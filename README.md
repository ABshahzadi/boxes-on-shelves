# 📦 Boxes on Shelves Optimization using NSGA-II

This project solves a multi-objective optimization problem to place boxes on shelves in a warehouse using the NSGA-II evolutionary algorithm.

## 🚀 Features
- Minimize the number of used shelves
- Minimize free space on shelves
- Handles box rotations (orientations)
- Supports different bay and shelf configurations
- Visualizes Pareto front and efficient frontier

## 🗂️ Project Structure
- `main.py`: Main script to run the optimization
- `data/`: Contains `products.txt`, `shelves.txt`, and `bay2.txt`
- `results/`: Stores output plots

## 🛠️ How to Run

Install dependencies:

```bash
pip install deap matplotlib
then run main