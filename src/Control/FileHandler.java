/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Control;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Asus
 */
public class FileHandler {
    
    private static String CURRENT_DIR;
    public static int NUM_FILES;
    public static Map<String, List<String>> LABELS = new HashMap<>();
    
    public static File[] readDirectoryContent(String path) {
        final File directory = new File(path);
        return directory.listFiles();
    }
    
    public static void readRecursive(String path) {
        
        // membaca file-file pada folder yang dipilih
        File[] entries = FileHandler.readDirectoryContent(path);
        for (File entry : entries) {
            if (entry.isDirectory()) {
                FileHandler.CURRENT_DIR = entry.getName();
                FileHandler.LABELS.put(FileHandler.CURRENT_DIR, 
                        new ArrayList<String>());
                FileHandler.readRecursive(path + "/" + entry.getName());
            }
            else {
                List<String> filenames = FileHandler.LABELS.get(
                        FileHandler.CURRENT_DIR);
                filenames.add(entry.getName());
                FileHandler.LABELS.put(FileHandler.CURRENT_DIR, filenames);
                FileHandler.NUM_FILES++;
            }
        }
        
    }
    
    public static void read(String path) {
        FileHandler.NUM_FILES = 0;
        FileHandler.LABELS = new HashMap<>();
        FileHandler.readRecursive(path);
    }
}
