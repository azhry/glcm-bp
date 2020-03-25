/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import Control.MathFx;

/**
 *
 * @author Asus
 */
public class Loss {
    public static double[] crossEntropy2(double[] actual, double[] predicted) {
        double[] losses = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            losses[i] = (actual[i] - predicted[i]) / (double)actual.length;
        }
        return losses;
    }
    
    public static double crossEntropy(double[] actual, double[] predicted) {
        double loss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            loss += ((actual[i] * Math.log(predicted[i])) + 
                    ((1 - actual[i]) * Math.log(1 - predicted[i])));
        }
        loss *= -(1.0 / (double)actual.length);
        return loss;
    }
    
    public static double error(double[] actual, double[] predicted) {
        double error = 0.0;
        int actualIndex = MathFx.maxIndex(actual);
        return - Math.log(predicted[actualIndex]) / (double)actual.length;
    }
    
    public static double crossEntropyDerivative(double actual, 
            double predicted) {
        double loss = (-1 * ((actual * (1 / predicted)) + 
                    (1 - actual) * (1 / (1 - predicted))));
        return loss;
    }
}
