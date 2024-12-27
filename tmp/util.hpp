// util.hpp

#ifndef UTIL_HPP
#define UTIL_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<int>>;

// Function to read a matrix from a file
inline Matrix readFromFile(const std::string &filename) {
  Matrix matrix;
  std::ifstream inFile(filename);
  if (inFile.is_open()) {
    std::string line;
    while (std::getline(inFile, line)) {
      std::istringstream iss(line);
      std::vector<int> row;
      int value;
      while (iss >> value) {
        row.push_back(value);
      }
      matrix.push_back(row);
    }
    inFile.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
  return matrix;
}

// Function to save a matrix to a file
inline void saveToFile(const Matrix &matrix, const std::string &filename) {
  std::ofstream outFile(filename);
  if (outFile.is_open()) {
    for (const auto &row : matrix) {
      for (const auto &elem : row) {
        outFile << elem << " ";
      }
      outFile << "\n";
    }
    outFile.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

#endif  // UTIL_HPP