/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

/**
 *
 * @author Asus
 */
public class Activation {
    
    public static double sigmoid(double n) {
        return 1 / (1 + Math.pow(Math.E, -n));
    }
    
    public static double sigmoidDerivative(double n) {
        double sigmoid = Activation.sigmoid(n);
        return sigmoid * (1.0 - sigmoid);
    }
    
    public static double[] sigmoidDerivatives(double[] n) {
        double[] results = new double[n.length];
        for (int i = 0; i < n.length; i++) {
            results[i] = Activation.sigmoidDerivative(n[i]);
        }
        return results;
    }
    
    public static double[] softmax(double[] n) {
        double[] result = new double[n.length];
        double sum = 0.0;
        for (double i : n) {
            sum += Math.pow(Math.E, i);
        }
        for (int i = 0; i < n.length; i++) {
            result[i] = Math.pow(Math.E, n[i]) / sum;
        }
        return result;
    }
    
    public static double[] softmaxDerivative(double[] n) {
        double[] output = Activation.softmax(n);
        double[] result = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            result[i] = output[i] * (1.0 - output[i]);
        }
        return result;
    }
    
    public static double relu(double n) {
        return Math.max(0, n);
    }
    
    public static double reluDerivative(double n) {
        return n > 0 ? 1 : 0;
    }
    
}
