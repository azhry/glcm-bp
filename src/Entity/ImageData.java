/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Entity;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Asus
 */
public class ImageData {
    
    private final String path;
    private String filename;
    private int width;
    private int height;
    private BufferedImage img;
    private int[][] pixels;
    private int[][] reds;
    private int[][] greens;
    private int[][] blues;
    private int[][] alphas;
    private int[][] greys;
    private String label;
    
    
    public ImageData(String path, String label) {
        this.path = path;
        this.label = label;
        this.img = this.readFile();
    }
    
    private BufferedImage readFile() {
        try {
            File file = new File(this.path);
            BufferedImage img = ImageIO.read(file);
            this.width = img.getWidth();
            this.height = img.getHeight();
            return img;
        } catch (IOException ex) {
            Logger.getLogger(ImageData.class.getName())
                    .log(Level.SEVERE, null, ex);
        }
        
        return null;
    }
    
    public String getLabel() {
        return this.label;
    }
    
    public String getPath() {
        return this.path;
    }
    
    public String getFilename() {
        return this.filename;
    }
    
    public void setFilename(String filename) {
        this.filename = filename;
    }
    
    public void logGreyPixels() {
        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                System.out.print(this.greys[i][j] + " ");
            }
            System.out.println();
        }
    }
    
    public int[][] readPixels() {
        this.pixels = new int[this.height][this.width];
        this.reds = new int[this.height][this.width];
        this.greens = new int[this.height][this.width];
        this.blues = new int[this.height][this.width];
        this.alphas = new int[this.height][this.width];
        this.greys = new int[this.height][this.width];
        
        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                int p = this.img.getRGB(i, j);
                this.pixels[i][j] = p;
                this.reds[i][j] = this.getRed(p);
                this.greens[i][j] = this.getGreen(p);
                this.blues[i][j] = this.getBlue(p);
                this.alphas[i][j] = this.getAlpha(p);
                this.greys[i][j] = this.getGrey(this.reds[i][j], 
                        this.greens[i][j], this.blues[i][j]);
            }
        }
        
        return this.pixels;
    }
    
    public int[][] getGreyPixels() {
        return this.greys;
    }
    
    public int getWidth() {
        return this.width;
    }
    
    public int getHeight() {
        return this.height;
    }
    
    private int getAlpha(int pixel) {
        return (pixel >> 24) & 0xff;
    }
    
    private int getRed(int pixel) {
        return (pixel >> 16) & 0xff;
    }
    
    private int getGreen(int pixel) {
        return (pixel >> 8) & 0xff;
    }
    
    private int getBlue(int pixel) {
        return pixel & 0xff;
    }
    
    private int getGrey(int red, int green, int blue) {
        return (int)(0.2989 * red + 0.5870 * green + 0.1141 * blue);
    }
}
