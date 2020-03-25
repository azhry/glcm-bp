/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Entity;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Asus
 */
public class CoOccurenceMatrix {
    
    private final ImageData img;
    private final int[][] greyPixels;
    private final int width;
    private final int height;
    private int maxPixel;
    private int minPixel;
    
    public CoOccurenceMatrix(ImageData img) {
        this.img = img;
        this.img.readPixels();
        this.greyPixels = this.img.getGreyPixels();
        this.width = this.img.getWidth();
        this.height = this.img.getHeight();
        this.setMinMaxPixel();
    }
    
    private void setMinMaxPixel() {
        this.maxPixel = Integer.MIN_VALUE;
        this.minPixel = Integer.MAX_VALUE;
        
        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                if (this.greyPixels[i][j] > this.maxPixel) {
                    this.maxPixel = this.greyPixels[i][j];
                }
                if (this.greyPixels[i][j] < this.minPixel) {
                    this.minPixel = this.greyPixels[i][j];
                }
            }
        }
    }
    
    public double[][] createCoOccurences() {
        int[][] matrix_0 = this.createMatrix(0);
        int[][] matrix_45 = this.createMatrix(45);
        int[][] matrix_90 = this.createMatrix(90);
        int[][] matrix_135 = this.createMatrix(135);
        
        int[][] matrix_0_t = this.createTransposeMatrix(matrix_0);
        int[][] matrix_45_t = this.createTransposeMatrix(matrix_45);
        int[][] matrix_90_t = this.createTransposeMatrix(matrix_90);
        int[][] matrix_135_t = this.createTransposeMatrix(matrix_135);
        
        int[][] matrix_0_symmetric = this.matrixAddition(matrix_0, matrix_0_t);
        int[][] matrix_45_symmetric = this.matrixAddition(matrix_45, matrix_45_t);
        int[][] matrix_90_symmetric = this.matrixAddition(matrix_90, matrix_90_t);
        int[][] matrix_135_symmetric = this.matrixAddition(matrix_135, matrix_135_t);
        
        double[][] matrix_0_normalized = this.normalizeMatrix(matrix_0_symmetric);
        double[][] matrix_45_normalized = this.normalizeMatrix(matrix_45_symmetric);
        double[][] matrix_90_normalized = this.normalizeMatrix(matrix_90_symmetric);
        double[][] matrix_135_normalized = this.normalizeMatrix(matrix_135_symmetric);
    
        double[][] coOccurenceMatrix = 
                this.createCoOccurenceMatrix(matrix_0_normalized, 
                        matrix_45_normalized, matrix_90_normalized, 
                        matrix_135_normalized);
        
        return coOccurenceMatrix;
    }
    
    public Map<String, Double> calculateFeatures(double[][] coOccurenceMatrix) {
        double asm = 0.0;
        double contrast = 0.0;
        double idm = 0.0;
        double entropy = 0.0;
        double mui = 0.0;
        double muj = 0.0;
        double sigmai = 0.0;
        double sigmaj = 0.0;
        double correlation = 0.0;
        
        for (int i = 0; i < coOccurenceMatrix.length; i++) {
            for (int j = 0; j < coOccurenceMatrix[i].length; j++) {
                asm += coOccurenceMatrix[i][j];
                contrast += Math.pow((i - j), 2) * coOccurenceMatrix[i][j];
                idm += ((1 - Math.pow(i - j, 2)) == 0 ? 0 : (coOccurenceMatrix[i][j]) / (1 - Math.pow(i - j, 2)));
                entropy += (coOccurenceMatrix[i][j] == 0 ? 0 : (coOccurenceMatrix[i][j] * Math.log(coOccurenceMatrix[i][j])));
                mui += (i * coOccurenceMatrix[i][j]);
                muj += (j * coOccurenceMatrix[i][j]);
            }
        }
        
        asm = Math.pow(asm, 2);
        entropy *= -1;
        
        for (int i = 0; i < coOccurenceMatrix.length; i++) {
            for (int j = 0; j < coOccurenceMatrix[i].length; j++) {
                sigmai += (coOccurenceMatrix[i][j] * Math.pow(i - mui, 2));
                sigmaj += (coOccurenceMatrix[i][j] * Math.pow(j - muj, 2));
            }
        }
        
        for (int i = 0; i < coOccurenceMatrix.length; i++) {
            for (int j = 0; j < coOccurenceMatrix[i].length; j++) {
                correlation += (((i - mui) * (j - muj) * coOccurenceMatrix[i][j]) / (sigmai * sigmaj));
            }
        }
        
        Map<String, Double> features = new HashMap<>();
        features.put("asm", asm);
        features.put("contrast", contrast);
        features.put("idm", idm);
        features.put("entropy", entropy);
        features.put("correlation", correlation);
        
        double lowerBound = Integer.MAX_VALUE;
        double upperBound = Integer.MIN_VALUE;
        
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            if (entry.getValue() < lowerBound) {
                lowerBound = entry.getValue();
            }
            
            if (entry.getValue() > upperBound) {
                upperBound = entry.getValue();
            }
        }
        
        Map<String, Double> normalizedFeatures = new HashMap<>();
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            double normalizedValue = ((0.8 * (entry.getValue() - lowerBound)) / (upperBound - lowerBound)) + 0.1;
            normalizedFeatures.put(entry.getKey(), normalizedValue);
        }
        
        return normalizedFeatures;
    }
    
    private double[][] createCoOccurenceMatrix(double[][] m0, double[][] m45, 
            double[][] m90, double[][] m135) {
        double[][] coOccurenceMatrix = new double[this.maxPixel + 1][this.maxPixel + 1];
        for (int i = 0; i < coOccurenceMatrix.length; i++) {
            for (int j = 0; j < coOccurenceMatrix[i].length; j++) {
                coOccurenceMatrix[i][j] = 
                        (m0[i][j] + m45[i][j] + m90[i][j] + m135[i][j]) / 4;
            }
        }
        return coOccurenceMatrix;
    }
    
    private double[][] normalizeMatrix(int[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];
        double total = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                total += (double)matrix[i][j];
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[i][j] = (double)matrix[i][j] / total;
            }
        }
        return result;
    }
    
    private int[][] matrixAddition(int[][] a, int[][] b) {
        int[][] result = new int[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    
    private int[][] createTransposeMatrix(int[][] matrix) {
        int[][] transpose = new int[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                transpose[i][j] = matrix[j][i];
            }
        }
        return transpose;
    }
    
    private int[][] createMatrix(int degree) {
        
        int matrix[][] = new int[this.maxPixel + 1][this.maxPixel + 1];
        switch (degree) {
            case 45:
                for (int i = 1; i < this.height; i++) {
                    for (int j = 0; j < this.width - 1; j++) {
                        int row = this.greyPixels[i][j];
                        int col = this.greyPixels[i - 1][j + 1];
                        matrix[row][col]++;
                    }
                }
                break;
            
            case 90:
                for (int i = 1; i < this.height; i++) {
                    for (int j = 0; j < this.width; j++) {
                        int row = this.greyPixels[i][j];
                        int col = this.greyPixels[i - 1][j];
                        matrix[row][col]++;
                    }
                }
                break;
                
            case 135:
                for (int i = 1; i < this.height; i++) {
                    for (int j = 1; j < this.width; j++) {
                        int row = this.greyPixels[i][j];
                        int col = this.greyPixels[i - 1][j - 1];
                        matrix[row][col]++;
                    }
                }
                break;
                
            default:
                for (int i = 0; i < this.height; i++) {
                    for (int j = 0; j < this.width - 1; j++) {
                        int row = this.greyPixels[i][j];
                        int col = this.greyPixels[i][j + 1];
                        matrix[row][col]++;
                    }
                }
                break;
        }
        
        return matrix;
    }
    
    public void logMatrix(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}
