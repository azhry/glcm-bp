/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Boundary;

import Control.FileHandler;
import Entity.CoOccurenceMatrix;
import Entity.ImageData;
import NeuralNetwork.ConfusionMatrix;
import NeuralNetwork.NeuralNetwork;
import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingWorker;
import javax.swing.table.DefaultTableModel;

/**
 *
 * @author Asus
 */
public class Main extends javax.swing.JFrame {

    private List<ImageData> data = new ArrayList<>();
    private List<CoOccurenceMatrix> comatrices = new ArrayList<>();
    private List<Map<String, Double>> features = new ArrayList<>();
    private List<ImageData> testData = new ArrayList<>();
    private List<CoOccurenceMatrix> testComatrices = new ArrayList<>();
    private List<Map<String, Double>> testFeatures = new ArrayList<>();
    private List<OutputNeuronLog> rowLog = new ArrayList<>();
    private boolean shuffled = false;
    
    /**
     * Creates new form Main
     */
    public Main() {
        initComponents();
    }
    
    public void displayResultTable(int[] predictedLabels) {
        int size = this.testData.size();
        Object[] labels = FileHandler.LABELS.keySet().toArray();
        String data[][] = new String[size][];    
        String column[] = {"No", "Filename", "Actual Label", "Predicted Label"};
        for (int i = 0; i < size; i++) {
            data[i] = new String[]{String.valueOf((i + 1)), 
                this.testData.get(i).getFilename(), 
                this.testData.get(i).getLabel(),
                String.valueOf(labels[predictedLabels[i]])};
        }
        

        JTable table = new JTable(data, column);
        JDialog dialog = new JDialog();
        dialog.setLocationRelativeTo(null);
        dialog.setLayout(new BorderLayout());
        dialog.add(new JScrollPane(table, 
              JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, 
              JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED),
              BorderLayout.CENTER);
        dialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
        dialog.pack();
        dialog.setVisible(true);
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        loadDataButton = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        featuresTable = new javax.swing.JTable();
        neuralNetworkLossChart = new javax.swing.JLabel();
        jScrollPane2 = new javax.swing.JScrollPane();
        nnResultTable = new javax.swing.JTable();
        learningRateField = new javax.swing.JTextField();
        epochField = new javax.swing.JSpinner();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jButton1 = new javax.swing.JButton();
        classifiedRatio = new javax.swing.JLabel();
        neuralNetworkProgressBar = new javax.swing.JProgressBar();
        outputNeuronLogPanel = new javax.swing.JPanel();
        randomizeCheckBox = new javax.swing.JCheckBox();
        loadDataTestingButton = new javax.swing.JButton();
        overallAccuracyLabel = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        loadDataButton.setText("Load Data Training");
        loadDataButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadImageData(evt);
            }
        });

        featuresTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null}
            },
            new String [] {
                "Filename", "ASM", "Contrast", "IDM", "Entropy", "Correlation", "Class"
            }
        ));
        jScrollPane1.setViewportView(featuresTable);

        neuralNetworkLossChart.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        neuralNetworkLossChart.setIcon(new javax.swing.ImageIcon(getClass().getResource("/Icon/icons8-system-task-100.png"))); // NOI18N
        neuralNetworkLossChart.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);

        nnResultTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {"Accuracy", null},
                {"Precision", null},
                {"Recall", null},
                {"F-Measure", null}
            },
            new String [] {
                "", "Result"
            }
        ) {
            boolean[] canEdit = new boolean [] {
                false, false
            };

            public boolean isCellEditable(int rowIndex, int columnIndex) {
                return canEdit [columnIndex];
            }
        });
        jScrollPane2.setViewportView(nnResultTable);

        learningRateField.setText("0.7");

        epochField.setValue(1000);

        jLabel1.setText("Learning Rate");

        jLabel2.setText("Epoch");

        jButton1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/Icon/icons8-play-24.png"))); // NOI18N
        jButton1.setText("Run Neural Network");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                runNeuralNetwork(evt);
            }
        });

        classifiedRatio.setText("0/0");

        neuralNetworkProgressBar.setString("");
        neuralNetworkProgressBar.setStringPainted(true);

        outputNeuronLogPanel.setLayout(new javax.swing.BoxLayout(outputNeuronLogPanel, javax.swing.BoxLayout.Y_AXIS));

        randomizeCheckBox.setText("Randomize");
        randomizeCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                randomizeCheckBoxActionPerformed(evt);
            }
        });

        loadDataTestingButton.setText("Load Data Testing");
        loadDataTestingButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadDataTesting(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(loadDataButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(loadDataTestingButton, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 508, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(26, 26, 26)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(randomizeCheckBox)
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                            .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                            .addComponent(learningRateField, javax.swing.GroupLayout.PREFERRED_SIZE, 97, javax.swing.GroupLayout.PREFERRED_SIZE))
                                        .addGap(45, 45, 45)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(jLabel2)
                                            .addComponent(epochField, javax.swing.GroupLayout.PREFERRED_SIZE, 92, javax.swing.GroupLayout.PREFERRED_SIZE)))))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(39, 39, 39)
                                .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 211, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(132, 132, 132)
                                .addComponent(classifiedRatio))
                            .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 270, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(overallAccuracyLabel))
                        .addGap(35, 35, 35)
                        .addComponent(outputNeuronLogPanel, javax.swing.GroupLayout.DEFAULT_SIZE, 217, Short.MAX_VALUE)
                        .addGap(24, 24, 24))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                            .addComponent(neuralNetworkProgressBar, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, 511, Short.MAX_VALUE)
                            .addComponent(neuralNetworkLossChart, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(97, 97, 97)
                        .addComponent(loadDataButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(loadDataTestingButton)
                        .addGap(0, 0, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jScrollPane1)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(neuralNetworkLossChart, javax.swing.GroupLayout.PREFERRED_SIZE, 196, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(neuralNetworkProgressBar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(outputNeuronLogPanel, javax.swing.GroupLayout.PREFERRED_SIZE, 268, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                        .addGap(0, 0, Short.MAX_VALUE)
                                        .addComponent(overallAccuracyLabel)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 105, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                            .addComponent(jLabel1)
                                            .addComponent(jLabel2))
                                        .addGap(8, 8, 8)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                            .addComponent(learningRateField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                            .addComponent(epochField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(randomizeCheckBox)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addComponent(jButton1)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addComponent(classifiedRatio)))))))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    // - loadImageData(evt : ActionEvent) : void
    private void loadImageData(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadImageData
        
        // memunculkan kotak dialog - start
        JFileChooser chooser = new JFileChooser();
        chooser.setCurrentDirectory(new java.io.File("./data"));
        chooser.setDialogTitle("Select Dataset Folder");
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // milih folder
        chooser.setAcceptAllFileFilterUsed(false);
        // memunculkan kotak dialog - end
        
        if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            // kode program yg dieksekusi ketika telah memilih file/folder
            String directory = chooser.getSelectedFile().toString();
            FileHandler.read(directory); // membaca isi direktori
            List<Map<String, Double>> dataFeatures = new ArrayList<>();
            List<Map<String, String>> metadata = new ArrayList<>();
            
            this.data = new ArrayList<>();
            this.comatrices = new ArrayList<>();
            this.features = new ArrayList<>();
            
            for (Map.Entry<String, List<String>> ent: 
                FileHandler.LABELS.entrySet()) {
                
                String path = directory + "/" + ent.getKey();
                for (String filename : ent.getValue()) {
                    System.out.println(filename);
                    ImageData img = new ImageData(path + "/" + filename, ent.getKey());
                    img.setFilename(filename);
                    img.readPixels(); // membaca pixel gambar
                    CoOccurenceMatrix comatrix = new CoOccurenceMatrix(img);
                    double[][] matrix = comatrix.createCoOccurences(); // membuat co-occurence matrix
                    Map<String, Double> featuresMatrix = comatrix.calculateFeatures(matrix);
                    dataFeatures.add(featuresMatrix);
                    
                    this.data.add(img);
                    this.comatrices.add(comatrix);
                    this.features.add(featuresMatrix);
                    
                    // menyimpan nama file dan class gambar
                    Map<String, String> meta = new HashMap<>();
                    meta.put("filename", filename);
                    meta.put("class", ent.getKey());
                    metadata.add(meta);
                }
                
            }
            
            DefaultTableModel model = (DefaultTableModel)this.featuresTable.getModel();
            model.setColumnCount(7);
            model.setRowCount(dataFeatures.size());
            
            // menampilkan hasil pada table
            for (int i = 0; i < dataFeatures.size(); i++) {
                Map<String, Double> featuresMatrix = dataFeatures.get(i);
                Map<String, String> meta = metadata.get(i);
                model.setValueAt(meta.get("filename"), i, 0);
                model.setValueAt(featuresMatrix.get("asm"), i, 1);
                model.setValueAt(featuresMatrix.get("contrast"), i, 2);
                model.setValueAt(featuresMatrix.get("idm"), i, 3);
                model.setValueAt(featuresMatrix.get("entropy"), i, 4);
                model.setValueAt(featuresMatrix.get("correlation"), i, 5);
                model.setValueAt(meta.get("class"), i, 6);
            }
        } 
        else {
            System.out.println("No Selection ");
        }
    }//GEN-LAST:event_loadImageData

    private void runNeuralNetwork(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runNeuralNetwork
        if (this.comatrices.size() <= 0) {
            JOptionPane.showMessageDialog(null, 
                    "Anda harus memuat data latih terlebih dahulu", "Error", 
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (this.testComatrices.size() <= 0) {
            JOptionPane.showMessageDialog(null, 
                    "Anda harus memuat data uji terlebih dahulu", "Error", 
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (this.learningRateField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(null, 
                    "Anda harus memasukkan nilai learning rate", "Error", 
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        int epoch = Integer.parseInt(String.valueOf(this.epochField.getValue()));
        double learningRate = Double.parseDouble(this.learningRateField.getText());
        double splitRatio = 0.7;
        
        if (learningRate <= 0 || learningRate > 1) {
            JOptionPane.showMessageDialog(null, 
                    "Nilai learning rate hanya diizinkan antara 0 sampai 1", "Error", 
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (epoch < 100 || epoch > 1500) {
            JOptionPane.showMessageDialog(null, 
                    "Nilai epoch hanya diizinkan antara 100 sampai 1500", "Error", 
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // menginisialisasi panel nilai output neuron
        this.rowLog = new ArrayList<>();
        this.outputNeuronLogPanel.removeAll();
        Object[] labels = FileHandler.LABELS.keySet().toArray();
        Map<String, double[]> encodedLabels = new HashMap<>();
        
        // mengubah class gambar dari string menjadi angka
        for (int i = 0; i < labels.length; i++) {
            double[] encoded = new double[labels.length];
            encoded[i] = 1.0;
            encodedLabels.put((String)labels[i], encoded);
            
            OutputNeuronLog log = new OutputNeuronLog();
            log.labelText.setText((String)labels[i]);
            log.valueText.setText("0.00");
            this.rowLog.add(log);
        }
        
        for (OutputNeuronLog log : this.rowLog) {
            this.outputNeuronLogPanel.add(log);
        }
        
        List<double[]> features = new ArrayList<>();
        List<double[]> classes = new ArrayList<>();
        
        boolean randomize = this.randomizeCheckBox.isSelected();
        
        // shuffle urutan data secara acak jika checkbox randomize dicentang
        if (!this.shuffled || randomize) {
            this.shuffled = true;
            Collections.shuffle(this.data);
            Collections.shuffle(this.comatrices);
        }
        
        // menampung features(co-occurence matrix) dan class ke dalam array
        for (CoOccurenceMatrix comatrix : this.comatrices) {
            features.add(comatrix.getFeatures());
            classes.add(encodedLabels.get(comatrix.getImageData().getLabel()));
        }
        
        double[][] finalFeatures = new double[features.size()][];
        double[][] finalClasses = new double[classes.size()][];
        for (int i = 0; i < features.size(); i++) {
            finalFeatures[i] = features.get(i);
            finalClasses[i] = classes.get(i);
        }
        
        // menginisialisasi object NeuralNetwork dengan nilai-nilai yang di-inputkan
        NeuralNetwork nn = new NeuralNetwork(finalFeatures, finalClasses, 
                labels.length + 2, learningRate, epoch, splitRatio);
        
        List<double[]> testFeatures = new ArrayList<>();
        List<double[]> testClasses = new ArrayList<>();
        
        // menampung features(co-occurence matrix) dan class test ke dalam array
        for (CoOccurenceMatrix comatrix : this.testComatrices) {
            testFeatures.add(comatrix.getFeatures());
            testClasses.add(encodedLabels.get(comatrix.getImageData().getLabel()));
        }
        
        double[][] finalTestFeatures = new double[testFeatures.size()][];
        double[][] finalTestClasses = new double[testClasses.size()][];
        for (int i = 0; i < testFeatures.size(); i++) {
            finalTestFeatures[i] = testFeatures.get(i);
            finalTestClasses[i] = testClasses.get(i);
        }
        
        nn.setTestData(finalTestFeatures, finalTestClasses);
        
        // menjalankan NeuralNetwork dengan worker
        new RunNeuralNetworkWorker(nn, this.neuralNetworkProgressBar, 
                this.neuralNetworkLossChart, this.rowLog, this.classifiedRatio, 
                this.nnResultTable, this.overallAccuracyLabel)
                .execute();
    }//GEN-LAST:event_runNeuralNetwork

    private void randomizeCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_randomizeCheckBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_randomizeCheckBoxActionPerformed

    private void loadDataTesting(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadDataTesting
        // memunculkan kotak dialog - start
        JFileChooser chooser = new JFileChooser();
        chooser.setCurrentDirectory(new java.io.File("./data"));
        chooser.setDialogTitle("Select Dataset Folder");
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // milih folder
        chooser.setAcceptAllFileFilterUsed(false);
        // memunculkan kotak dialog - end
        
        if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            // kode program yg dieksekusi ketika telah memilih file/folder
            String directory = chooser.getSelectedFile().toString();
            FileHandler.read(directory); // membaca isi direktori
            List<Map<String, Double>> dataFeatures = new ArrayList<>();
            List<Map<String, String>> metadata = new ArrayList<>();
            
            this.testData = new ArrayList<>();
            this.testComatrices = new ArrayList<>();
            this.testFeatures = new ArrayList<>();
            
            for (Map.Entry<String, List<String>> ent: 
                FileHandler.LABELS.entrySet()) {
                
                String path = directory + "/" + ent.getKey();
                for (String filename : ent.getValue()) {
                    System.out.println(filename);
                    ImageData img = new ImageData(path + "/" + filename, ent.getKey());
                    img.setFilename(filename);
                    img.readPixels(); // membaca pixel gambar
                    CoOccurenceMatrix comatrix = new CoOccurenceMatrix(img); // membuat co-occurence matrix
                    double[][] matrix = comatrix.createCoOccurences();
                    Map<String, Double> featuresMatrix = comatrix.calculateFeatures(matrix);
                    dataFeatures.add(featuresMatrix);
                    
                    this.testData.add(img);
                    this.testComatrices.add(comatrix);
                    this.testFeatures.add(featuresMatrix);
                    
                    // menyimpan nama file dan class gambar
                    Map<String, String> meta = new HashMap<>();
                    meta.put("filename", filename);
                    meta.put("class", ent.getKey());
                    metadata.add(meta);
                }
                
            }
            
            DefaultTableModel model = (DefaultTableModel)this.featuresTable.getModel();
            model.setColumnCount(7);
            model.setRowCount(dataFeatures.size());
            
            // menampilkan hasil pada table
            for (int i = 0; i < dataFeatures.size(); i++) {
                Map<String, Double> featuresMatrix = dataFeatures.get(i);
                Map<String, String> meta = metadata.get(i);
                model.setValueAt(meta.get("filename"), i, 0);
                model.setValueAt(featuresMatrix.get("asm"), i, 1);
                model.setValueAt(featuresMatrix.get("contrast"), i, 2);
                model.setValueAt(featuresMatrix.get("idm"), i, 3);
                model.setValueAt(featuresMatrix.get("entropy"), i, 4);
                model.setValueAt(featuresMatrix.get("correlation"), i, 5);
                model.setValueAt(meta.get("class"), i, 6);
            }
        } 
        else {
            System.out.println("No Selection ");
        }
    }//GEN-LAST:event_loadDataTesting
    
    class RunNeuralNetworkWorker extends SwingWorker {

        private NeuralNetwork nn;
        private final JProgressBar progressBar;
        private final javax.swing.JLabel lossChart;
        private final javax.swing.JLabel classifiedRatioText;
        private final javax.swing.JLabel overallAccuracyText;
        private final javax.swing.JTable nnResultTable;
        private final List<OutputNeuronLog> logs;
        
        public RunNeuralNetworkWorker(NeuralNetwork nn, 
                JProgressBar progressBar, javax.swing.JLabel lossChart, 
                List<OutputNeuronLog> logs, 
                javax.swing.JLabel classifiedRatioText, 
                javax.swing.JTable nnResultTable, 
                javax.swing.JLabel overallAccuracyText) {
            this.nn = nn;
            this.progressBar = progressBar;
            this.lossChart = lossChart;
            this.logs = logs;
            this.classifiedRatioText = classifiedRatioText;
            this.nnResultTable = nnResultTable;
            this.overallAccuracyText = overallAccuracyText;
        }
        
        @Override
        protected void done() {
           
            JOptionPane.showMessageDialog(null, 
                    "Neural Network process is done", "Done", 
                    JOptionPane.INFORMATION_MESSAGE);
            
            int[] predictedLabels = this.nn.getPredictedLabels();
            displayResultTable(predictedLabels);
            
            repaint();
        }
        
        @Override
        protected Object doInBackground() throws Exception {
            // mengeksekusi NeuralNetwork dan menyimpan hasilnya ke confusion matrix
            ConfusionMatrix cm = this.nn.fit(this.progressBar, this.lossChart, this.logs, 
                    this.classifiedRatioText, this.nnResultTable, this.overallAccuracyText);
            return null;
        }
        
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Main().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel classifiedRatio;
    private javax.swing.JSpinner epochField;
    private javax.swing.JTable featuresTable;
    private javax.swing.JButton jButton1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JTextField learningRateField;
    private javax.swing.JButton loadDataButton;
    private javax.swing.JButton loadDataTestingButton;
    private javax.swing.JLabel neuralNetworkLossChart;
    private javax.swing.JProgressBar neuralNetworkProgressBar;
    private javax.swing.JTable nnResultTable;
    private javax.swing.JPanel outputNeuronLogPanel;
    private javax.swing.JLabel overallAccuracyLabel;
    private javax.swing.JCheckBox randomizeCheckBox;
    // End of variables declaration//GEN-END:variables
}
