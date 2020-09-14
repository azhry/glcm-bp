/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import Boundary.OutputNeuronLog;
import Control.MathFx;
import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.ImageIcon;
import javax.swing.JProgressBar;
import javax.swing.table.DefaultTableModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 *
 * @author Asus
 */
public class NeuralNetwork {
    private final double learningRate;
    private final int EPOCH;
    
    private final int numInputNeuron;
    private final int numHiddenNeuron1;
    private final int numHiddenNeuron2;
    private final int numOutputNeuron;
    
    private Neuron[] inputNeurons;
    private Neuron[] hiddenNeurons1;
    private Neuron[] hiddenNeurons2;
    private Neuron[] outputNeurons;
    
    private double[][] inputHidden1Connections;
    private double[][] hidden1Hidden2Connections;
    private double[][] hidden2OutputConnections;
    
    private double[] deltaInput;
    private double[] deltaHidden1;
    private double[] deltaHidden2;
    
    private double[] crossEntropyDerivatives;
    private final double[][] data;
    private final double[][] target;
    private double[][] testData;
    private double[][] testTarget;
    
    private int[] predictedLabels;
    
    private double error;
    private List<Double> epochLoss;
    
    private ConfusionMatrix trainCm;
    private ConfusionMatrix testCm;
    
    public NeuralNetwork(double[][] data, double[][] target, 
            int numHiddens, double learningRate, int epoch, double splitRatio) {
        int numData = data.length;

        this.data = data;
        this.target = target;
        
        this.learningRate = learningRate;
        this.EPOCH = epoch;
        
        this.numInputNeuron = this.data[0].length;
        this.numHiddenNeuron1 = numHiddens;
        this.numHiddenNeuron2 = numHiddens;
        this.numOutputNeuron = this.target[0].length;
        
        this.initializeInputNeurons();
        this.initializeHiddenNeurons1();
        this.initializeHiddenNeurons2();
        this.initializeOutputNeurons();
    
        this.initializeInputHidden1Connections();
        this.initializeHidden1Hidden2Connections();
        this.initializeHidden2OutputConnections();
    }
    
    public void setTestData(double[][] data, double[][] target) {
        this.testData = data;
        this.testTarget = target;
        this.predictedLabels = new int[this.testData.length];
    }
    
    public int[] getPredictedLabels() {
        return this.predictedLabels;
    }
    
    public ConfusionMatrix fit(JProgressBar progressBar, 
            javax.swing.JLabel neuralNetworkLossChart, 
            List<OutputNeuronLog> logs, 
            javax.swing.JLabel classifiedRatioText, 
            javax.swing.JTable nnResultTable, 
            javax.swing.JLabel overallAccuracyLabel) {
        this.epochLoss = new ArrayList<>();
        int progress = 0;
        int currentProgress = 0;
        int maxProgress = this.EPOCH * this.data.length;
        int corrects = 0;
        int incorrects = 0;
        this.trainCm = new ConfusionMatrix();
        
        DefaultTableModel model = (DefaultTableModel)nnResultTable.getModel();
        model.setRowCount(4);
        model.setColumnCount(2);
        
        for (int e = 0; e < this.EPOCH; e++) {
            for (int i = 0; i < this.data.length; i++) {
                // melakukan feedforward
                this.feedforward(this.data[i]);
                
                List<Double> outputNeuronValues = new ArrayList<>();
                
                // set warna pada tampilan output neuron log
                for (int j = 0; j < this.numOutputNeuron; j++) {
                    logs.get(j).valueText
                            .setText(String.valueOf(
                                    this.outputNeurons[j].getMappedValue()));
                    logs.get(j).setBackground(Color.WHITE);
                    outputNeuronValues
                            .add(this.outputNeurons[j].getMappedValue());
                }
                
                // menentukan class yang terprediksi
                int maxPredictedIndex = MathFx.maxIndex(outputNeuronValues);
                List<Double> listTarget = new ArrayList<>();
                for (double t : this.target[i]) {
                    listTarget.add(t);
                }
                
                // menentukan class actual
                int maxActualIndex = MathFx.maxIndex(listTarget);
                
                // update confusion matrix dengan class terprediksi dan class actual
                this.trainCm.update(maxActualIndex, maxPredictedIndex);
                
                // menghitung jumlah klasifikasi benar dan salah
                if (maxActualIndex == maxPredictedIndex) {
                    logs.get(maxPredictedIndex)
                        .setBackground(Color.GREEN);
                    corrects++;
                }
                else {
                    logs.get(maxPredictedIndex)
                        .setBackground(Color.RED);
                    incorrects++;
                }
                
                classifiedRatioText.setText(corrects + "/" + incorrects + 
                        " (" + (incorrects - corrects) + ")");
                this.error = this.calculateError(this.target[i]);
                
                // melakukan backpropagation
                this.backpropagation(this.target[i]);
                currentProgress++;
                progress = (int)(((double)currentProgress / 
                        (double)maxProgress) * 100);
                progressBar.setValue(progress);
                progressBar.setString(progress + "%");
            }
            
            // menampilkan hasil ke dalam table
            model.setValueAt("(Train = " + Math.round(((this.trainCm.getAccuracy() * 100.0) / 100.0) 
                    * 100.0) + "%)", 0, 1);
            model.setValueAt("(Train = " + Math.round(((this.trainCm.getPrecision() * 100.0) / 100.0) 
                    * 100.0) + "%)", 1, 1);
            model.setValueAt("(Train = " + Math.round(((this.trainCm.getRecall() * 100.0) / 100.0) 
                    * 100.0) + "%)", 2, 1);
            model.setValueAt("(Train = " + Math.round(((this.trainCm.getF1score() * 100.0) / 100.0) 
                    * 100.0) + "%)", 3, 1);
            this.epochLoss.add(this.error);
            this.displayLossChart(neuralNetworkLossChart);
            
            if (e == this.EPOCH - 1) {
                break;
            }
            this.trainCm.reset();
        }
    
        // menyimpan bobot NeuralNetwork
        this.saveWeight(progressBar);
        
        // melakukan pengujian
        this.score(this.testData, this.testTarget, model, overallAccuracyLabel);
        return this.trainCm;
    }
    
    public void score(double[][] data, double[][] target, 
            DefaultTableModel model, javax.swing.JLabel overallAccuracyLabel) {
        
        // memuat bobot yang disimpan
        this.loadWeight();

        // inisialisasi Confusion Matrix pengujian
        this.testCm = new ConfusionMatrix();
        for (int i = 0; i < data.length; i++) {
            // melakukan feedforward
            this.feedforward(data[i]);
            
            List<Double> outputNeuronValues = new ArrayList<>();
            for (int j = 0; j < this.numOutputNeuron; j++) {
                outputNeuronValues.add(this.outputNeurons[j].getMappedValue());
            }
            
            // menentukan class terprediksi
            int maxPredictedIndex = MathFx.maxIndex(outputNeuronValues);
            List<Double> listTarget = new ArrayList<>();
            for (double t : target[i]) {
                listTarget.add(t);
            }
            
            this.predictedLabels[i] = maxPredictedIndex;

            // menentukan class actual
            int maxActualIndex = MathFx.maxIndex(listTarget);
            
            // update confusion matrix dengan class terprediksi dan class actual
            this.testCm.update(maxActualIndex, maxPredictedIndex);
            System.out.println(maxActualIndex + " " + maxPredictedIndex);
        }
        
        this.testCm.showMatrix();
        
        // menghitung akurasi total keseluruhan
        double totalAccuracy = (double)(this.trainCm.getTruePositive() + 
                this.testCm.getTruePositive()) / 
                (double)(this.trainCm.getTotalSamples() + 
                this.testCm.getTotalSamples());
        
        System.out.println(this.trainCm.getTruePositive());
        System.out.println(this.testCm.getTruePositive());
        System.out.println(this.testCm.getTotalSamples());
        System.out.println(this.trainCm.getTotalSamples());
        System.out.println(totalAccuracy);
        
        // menampilkan hasil pengujian ke dalam table
        model.setValueAt(model.getValueAt(0, 1) + ", (Test = " + 
                Math.round(((this.testCm.getAccuracy() * 100.0) / 100.0) 
                * 100.0) + "%)", 0, 1);
        model.setValueAt(model.getValueAt(1, 1) + ", (Test = " + 
                Math.round(((this.testCm.getPrecision() * 100.0) / 100.0) 
                * 100.0) + "%)", 1, 1);
        model.setValueAt(model.getValueAt(2, 1) + ", (Test = " + 
                Math.round(((this.testCm.getRecall() * 100.0) / 100.0) 
                * 100.0) + "%)", 2, 1);
        model.setValueAt(model.getValueAt(3, 1) + ", (Test = " + 
                Math.round(((this.testCm.getF1score() * 100.0) / 100.0) 
                * 100.0) + "%)", 3, 1);
        
        
        // menampilkan akurasi total keseluruhan
        overallAccuracyLabel.setText("(Overall Accuracy = " + 
                (totalAccuracy * 100.0) + "%)");
    }
    
    public void loadWeight() {
        try {
                
            JSONObject inputHidden1Weights = (JSONObject)new JSONParser()
                        .parse(new FileReader("InputHidden1Weights.json"));
            int numRows = inputHidden1Weights.size();
            int numCols = ((JSONObject)inputHidden1Weights.get("0")).size();

            for (int i = 0; i < numRows; i++) {
                numCols = ((JSONObject)inputHidden1Weights.get(String.valueOf(i))).size();
                for (int j = 0; j < numCols; j++) {
                    this.inputHidden1Connections[i][j] = 
                            (double)((JSONObject)(inputHidden1Weights
                                    .get(String.valueOf(i))))
                                    .get(String.valueOf(j));
                }
            }
            
            JSONObject hidden1Hidden2Weights = (JSONObject)new JSONParser()
                        .parse(new FileReader("Hidden1Hidden2Weights.json"));
            numRows = hidden1Hidden2Weights.size();
            numCols = ((JSONObject)hidden1Hidden2Weights.get("0")).size();
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    this.hidden1Hidden2Connections[i][j] = 
                            (double)((JSONObject)hidden1Hidden2Weights
                                    .get(String.valueOf(i)))
                                    .get(String.valueOf(j));
                }
            }
            
            JSONObject hidden2OutputWeights = (JSONObject)new JSONParser()
                        .parse(new FileReader("Hidden2OutputWeights.json"));
            numRows = hidden2OutputWeights.size();
            numCols = ((JSONObject)hidden2OutputWeights.get("0")).size();
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    this.hidden2OutputConnections[i][j] = 
                            (double)((JSONObject)hidden2OutputWeights
                                    .get(String.valueOf(i)))
                                    .get(String.valueOf(j));
                }
            }
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void saveWeight() {
        JSONObject inputHidden1Weights = new JSONObject();
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numInputNeuron; j++) {
                rows.put(j, this.inputHidden1Connections[i][j]);
            }
            inputHidden1Weights.put(i, rows);
        }
        
        JSONObject hidden1Hidden2Weights = new JSONObject();
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                rows.put(j, this.hidden1Hidden2Connections[i][j]);
            }
            hidden1Hidden2Weights.put(i, rows);
        }
        
        JSONObject hidden2OutputWeights = new JSONObject();
        for (int i = 0; i < this.numOutputNeuron; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                rows.put(j, this.hidden2OutputConnections[i][j]);
            }
            hidden2OutputWeights.put(i, rows);
        }
        
        try {
            FileWriter writer = new FileWriter("InputHidden1Weights.json");
            writer.write(inputHidden1Weights.toString());
            writer.flush();
            
            writer = new FileWriter("Hidden1Hidden2Weights.json");
            writer.write(hidden1Hidden2Weights.toString());
            writer.flush();
            
            writer = new FileWriter("Hidden2OutputWeights.json");
            writer.write(hidden2OutputWeights.toString());
            writer.flush();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public void saveWeight(JProgressBar progressBar) {
        progressBar.setString("Saving weights....");
        JSONObject inputHidden1Weights = new JSONObject();
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numInputNeuron; j++) {
                rows.put(j, this.inputHidden1Connections[i][j]);
            }
            inputHidden1Weights.put(i, rows);
        }
        
        JSONObject hidden1Hidden2Weights = new JSONObject();
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                rows.put(j, this.hidden1Hidden2Connections[i][j]);
            }
            hidden1Hidden2Weights.put(i, rows);
        }
        
        JSONObject hidden2OutputWeights = new JSONObject();
        for (int i = 0; i < this.numOutputNeuron; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                rows.put(j, this.hidden2OutputConnections[i][j]);
            }
            hidden2OutputWeights.put(i, rows);
        }
        
        try {
            FileWriter writer = new FileWriter("InputHidden1Weights.json");
            writer.write(inputHidden1Weights.toString());
            writer.flush();
            
            writer = new FileWriter("Hidden1Hidden2Weights.json");
            writer.write(hidden1Hidden2Weights.toString());
            writer.flush();
            
            writer = new FileWriter("Hidden2OutputWeights.json");
            writer.write(hidden2OutputWeights.toString());
            writer.flush();
            
            progressBar.setString("Weights Saved");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    
    public void displayLossChart(javax.swing.JLabel lossChart) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < this.epochLoss.size(); i++) {
            dataset.addValue(this.epochLoss.get(i), "Loss", 
                    String.valueOf((i + 1)));
        }
        JFreeChart chart = ChartFactory.createLineChart(
                "Error / Loss Overtime",
                "Time", "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false
        );
        
        BufferedImage imageChart = chart.createBufferedImage(
                lossChart.getWidth(), lossChart.getHeight());
        Image im = imageChart;
        lossChart.setIcon(new ImageIcon(im));
        
    }
    
    private double calculateError(double[] actual) {
        double[] predicted = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            predicted[i] = this.outputNeurons[i].getMappedValue();
        }
        return Loss.error(actual, predicted);
    }
    
    
    private void feedforward(double[] data) {
        this.calculateInputHidden1(data); // menghitung nilai dari input ke hidden1
        this.calculateHidden1Hidden2(); // menghitung nilai dari hidden1 ke hidden2
        this.calculateHidden2Output(); // menghitung nilai dari hidden2 ke output
    }
    
    private void backpropagation(double[] actual) {
        this.calculateOutputHidden2(actual);
    }
    
    private void calculateInputHidden1(double[] data) {
        for (int i = 0; i < this.numInputNeuron; i++) {
            this.inputNeurons[i].setValue(data[i]); // set nilai tiap neuron input
        }
        
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            double totalInputValue = 0.0;
            
            // menhitung total nilai tiap koneksi
            for (int j = 0; j < this.numInputNeuron; j++) {
                totalInputValue += (this.inputHidden1Connections[i][j] * 
                        this.inputNeurons[j].getValue());
            }
            
            // menghitung nilai dengan fungsi aktivasi sigmoid
            double mappedValue = Activation.sigmoid(totalInputValue + bias);
            
            // set nilai tiap neuron hidden1
            this.hiddenNeurons1[i].setValue(totalInputValue + bias);
            this.hiddenNeurons1[i].setMappedValue(mappedValue);
        }
    }
    
    private void calculateHidden1Hidden2() {
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            double totalInputValue = 0.0;
            
            // menhitung total nilai tiap koneksi
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                totalInputValue += (this.hidden1Hidden2Connections[i][j] * 
                        this.hiddenNeurons1[j].getMappedValue());
            }
            
            // menghitung nilai dengan fungsi aktivasi sigmoid
            double mappedValue = Activation.sigmoid(totalInputValue + bias);
            
            // set nilai tiap neuron hidden2
            this.hiddenNeurons2[i].setValue(totalInputValue + bias);
            this.hiddenNeurons2[i].setMappedValue(mappedValue);
        }
    }
    
    private double[] calculateHidden2Output() {
        
        double bias = MathFx.randUniform(1);
        double[] inputValues = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            double totalInputValue = 0.0;
            
            // menhitung total nilai tiap koneksi
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                totalInputValue += (this.hidden2OutputConnections[i][j] * 
                        this.hiddenNeurons2[j].getMappedValue());
            }
            inputValues[i] = totalInputValue + bias;
            this.outputNeurons[i].setValue(inputValues[i]);
        }
        
        // menghitung nilai dengan fungsi aktivasi softmax
        double[] mappedValues = Activation.softmax(inputValues);
        
        // set nilai tiap neuron output
        for (int i = 0; i < mappedValues.length; i++) {
            this.outputNeurons[i].setMappedValue(mappedValues[i]);
        }
        
        return mappedValues;
    }
    
    private void calculateOutputHidden2(double[] actual) {
        this.crossEntropyDerivatives = new double[this.numOutputNeuron];
        double[] outputNeuronValues = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            outputNeuronValues[i] = this.outputNeurons[i].getValue();
        }
        
        // menghitung loss dengan cross entropy
        double[] a3_delta = Loss.crossEntropy2(actual, outputNeuronValues);
        double[] z2_delta = new double[this.numHiddenNeuron2];
        for (int j = 0; j < this.hidden2OutputConnections[0].length; j++) {
            for (int k = 0; k < this.hidden2OutputConnections.length; k++) {
                
                z2_delta[j] += a3_delta[k] * 
                        this.hidden2OutputConnections[k][j];

            }
        }
        
        for (int i = 0; i < this.hidden2OutputConnections.length; i++) {
            for (int j = 0; j < this.hidden2OutputConnections[i].length; j++) {
                
                // update bobot koneksi hidden2 ke output
                this.hidden2OutputConnections[i][j] += 
                        this.learningRate * a3_delta[i] 
                        * this.hiddenNeurons2[j].getMappedValue();
                
            }
        }
        
        double[] hidden2NeuronValues = new double[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            hidden2NeuronValues[i] = this.hiddenNeurons2[i].getMappedValue();
        }
        
        // menghitung delta dengan fungsi turunan sigmoid
        double[] a2_delta = Activation.sigmoidDerivatives(hidden2NeuronValues);
        for (int i = 0; i < a2_delta.length; i++) {
            a2_delta[i] = a2_delta[i] * z2_delta[i];
        }
        
        double[] z1_delta = new double[this.numHiddenNeuron1];
        for (int i = 0; i < this.hidden1Hidden2Connections[0].length; i++) {
            for (int j = 0; j < this.hidden1Hidden2Connections.length; j++) {
                
                z1_delta[i] += a2_delta[j] * 
                        this.hidden1Hidden2Connections[j][i];
            }
        }
        
        for (int i = 0; i < this.hidden1Hidden2Connections.length; i++) {
            for (int j = 0; j < this.hidden1Hidden2Connections[i].length; j++) {
                // update bobot koneksi hidden1 ke hidden2
                this.hidden1Hidden2Connections[i][j] += 
                        this.learningRate * a2_delta[i] * 
                        this.hiddenNeurons1[j].getMappedValue();
                
            }
        }
        
        double[] hidden1NeuronValues = new double[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            hidden1NeuronValues[i] = this.hiddenNeurons1[i].getMappedValue();
        }
        
        // menghitung delta dengan fungsi turunan sigmoid
        double[] a1_delta = Activation.sigmoidDerivatives(hidden1NeuronValues);
        for (int i = 0; i < a1_delta.length; i++) {
            a1_delta[i] = a1_delta[i] * z1_delta[i];
        }
        
        for (int i = 0; i < this.inputHidden1Connections.length; i++) {
            for (int j = 0; j < this.inputHidden1Connections[i].length; j++) {
                // update bobot koneksi input ke hidden1
                this.inputHidden1Connections[i][j] += 
                        this.learningRate * a1_delta[i] *
                        this.inputNeurons[j].getValue();
                
            }
        }
    }
    
    private void calculateHidden2Hidden1() {
//        double[] sigmoidDerivatives = new double[this.numHiddenNeuron2];
        
//        for (int i = 0; i < this.numHiddenNeuron1; i++) {
//            this.deltaHidden1[i] = 0.0;
//            for (int j = 0; j < this.numHiddenNeuron2; j++) {
//                double sigmoidDerivatives = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons2[j].getValue());
//                
//                this.deltaHidden1[i] += this.deltaHidden2[j] * 
//                        this.hiddenNeurons1[i].getMappedValue() * 
//                        sigmoidDerivatives;
//                
//                this.hidden1Hidden2Connections[j][i] += (this.learningRate * 
//                        this.deltaHidden2[j] * 
//                        sigmoidDerivatives);
//            }
//        }
        
        
//        for (int i = 0; i < this.numHiddenNeuron2; i++) {
//            
//            sigmoidDerivatives[i] = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons2[i].getValue());
//            
//            for (int j = 0; j < this.numHiddenNeuron1; j++) {
//                this.hidden1Hidden2Connections[i][j] += 
//                        this.learningRate * (this.crossEntropyDerivatives[i] *
//                        sigmoidDerivatives[i] * 
//                        this.hiddenNeurons1[j].getMappedValue());
//            }
//            
//        }
    }
    
    private void calculateHidden1Input() {
//        for (int i = 0; i < this.numInputNeuron; i++) {
//            this.deltaInput[i] = 0.0;
//            for (int j = 0; j < this.numHiddenNeuron1; j++) {
//                double sigmoidDerivatives = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons1[j].getValue());
//                
//                this.deltaInput[i] += this.deltaHidden1[j] * 
//                        this.inputNeurons[i].getMappedValue() * 
//                        sigmoidDerivatives;
//                
//                this.inputHidden1Connections[j][i] += (this.learningRate * 
//                        this.deltaHidden1[j] * 
//                        sigmoidDerivatives);
//            }
//        }
        
//        double[] sigmoidDerivatives = new double[this.numHiddenNeuron1];
//        for (int i = 0; i < this.numHiddenNeuron1; i++) {
//            
//            sigmoidDerivatives[i] = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons1[i].getValue());
//            
//            for (int j = 0; j < this.numInputNeuron; j++) {
//                this.inputHidden1Connections[i][j] += 
//                        this.learningRate * (this.crossEntropyDerivatives[i] *
//                        sigmoidDerivatives[i] * 
//                        this.inputNeurons[j].getMappedValue());
//            }
//            
//        }
    }
    
    
    
    public double[][] initializeInputHidden1Connections() {
        this.inputHidden1Connections = 
                new double[this.numHiddenNeuron1][this.numInputNeuron];
        this.deltaInput = new double[this.numInputNeuron];
        this.deltaHidden1 = new double[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            for (int j = 0; j < this.numInputNeuron; j++) {
                this.inputHidden1Connections[i][j] = MathFx.randUniform(1);
            }
        }
        return this.inputHidden1Connections;
    }
    
    public double[][] initializeHidden1Hidden2Connections() {
        this.hidden1Hidden2Connections = 
                new double[this.numHiddenNeuron2][this.numHiddenNeuron1];
        this.deltaHidden2 = new double[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                this.hidden1Hidden2Connections[i][j] = MathFx.randUniform(1);
            }
        }
        return this.hidden1Hidden2Connections;
    }
    
    public double[][] initializeHidden2OutputConnections() {
        this.hidden2OutputConnections =
                new double[this.numOutputNeuron][this.numHiddenNeuron2];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                this.hidden2OutputConnections[i][j] = MathFx.randUniform(1);
            }
        }
        return this.hidden2OutputConnections;
    }
    
    private void initializeInputNeurons() {
        this.inputNeurons = new Neuron[this.numInputNeuron];
        for (int i = 0; i < this.numInputNeuron; i++) {
            this.inputNeurons[i] = new Neuron(Neuron.Type.INPUT);
        }
    }
    
    private void initializeHiddenNeurons1() {
        this.hiddenNeurons1 = new Neuron[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            this.hiddenNeurons1[i] = new Neuron(Neuron.Type.HIDDEN);
        }
    }
    
    private void initializeHiddenNeurons2() {
        this.hiddenNeurons2 = new Neuron[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            this.hiddenNeurons2[i] = new Neuron(Neuron.Type.HIDDEN);
        }
    }
    
    private void initializeOutputNeurons() {
        this.outputNeurons = new Neuron[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            this.outputNeurons[i] = new Neuron(Neuron.Type.OUTPUT);
        }
    }
}
