
package flappybird;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import javax.swing.*;
import javax.swing.Timer;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Ben Giamarino
 */
public class FlappyBird implements ActionListener, MouseListener
{
    //Height and width constants for window
    public final int WIDTH  = 1200;
    public final int HEIGHT = 600;
    
    //Initialize variables
    public static FlappyBird flappyBird;
    public Renderer renderer;
    public Rectangle bird;
    public ArrayList<Rectangle> columns;
    public Random rand;
    public int ticks, yMotion, score;
    public boolean gameOver, started; //Is the game over and has it started
    
    //Constructor
    public FlappyBird()
    {
        JFrame jframe = new JFrame();
        Timer timer   = new Timer(20, this);
        renderer      = new Renderer();
        bird          = new Rectangle(WIDTH/2 -10, HEIGHT/2 - 10, 20, 20);
        rand          = new Random();
        columns       = new ArrayList<>();

        //Build Game Environment
        jframe.add(renderer);
        jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jframe.setSize(WIDTH, HEIGHT);
        jframe.setResizable(false);
        jframe.setTitle("Flappy Bird");
        jframe.setVisible(true);
        jframe.addMouseListener(this);
         
        addColumn(true);
        addColumn(true);
        addColumn(true);
        addColumn(true);
        
        timer.start();
    }
    
    public void addColumn(boolean start)
    {
        int space  = 300;
        int width  = 100;
        int height = 50 + rand.nextInt(300);
        
        if(start)
        {
            //Creating a column with an upper and lower portion and a gap somewhere randomly in between
            columns.add(new Rectangle(WIDTH + width + columns.size() * 300, HEIGHT - height - 120, width, height));
            columns.add(new Rectangle(WIDTH + width + (columns.size() - 1) * 300, 0, width, HEIGHT - height - space));
        }
        else
        {
            columns.add(new Rectangle(columns.get(columns.size() - 1).x + 600, HEIGHT - height - 120, width, height));
            columns.add(new Rectangle(columns.get(columns.size() - 1).x, 0, width, HEIGHT - height - space));
        }
        
    }
    
    public void actionPerformed(ActionEvent e)
    {
        int speed = 10;
        
        ticks++;
        
        if(started)   
        {
            //Move the columns across the screen from right to left at a given speed
            for(int i = 0; i < columns.size(); i++)
            {
                Rectangle column = columns.get(i);

                if(column.x + column.width < 0)
                {
                    columns.remove(column);

                    if(column.y == 0)
                    {
                        addColumn(false);
                    }
                }
                else
                {
                    column.x -= speed;
                }
            }

            if(ticks % 2 == 0 && yMotion < 15)
            {
                yMotion += 2;
            }

            bird.y += yMotion;

            for(Rectangle column : columns)
            {
                if(column.intersects(bird))
                {
                    bird.x = column.x - bird.width;
                    gameOver = true;                    
                }
            }
            
            //Check collision of bird with ceiling or ground
            if(bird.y > HEIGHT - 120 || bird.y < 0)
            {
                gameOver = true;
            }
            
            if(gameOver)
            {
                bird.y = HEIGHT - 120 - bird.height;
            }
        }
        
        renderer.repaint();
    }
    
    public void jump()
    {
        if(gameOver)
        {
            bird = new Rectangle(WIDTH/2 -10, HEIGHT/2 - 10, 20, 20);
            columns.clear();
            
            addColumn(true);
            addColumn(true);
            addColumn(true);
            addColumn(true);
            
            gameOver = false;
            yMotion = 0;
            score = 0;
        }
        
        if(!started)
        {
            started = true;
        }
        else if(!gameOver)
        {
            if(yMotion > 0)
            {
                yMotion = 0;
            }
            
            yMotion -= 10;
        }
    }
    
    public void paintColumn(Graphics g, Rectangle column)
    {
        g.setColor(Color.green.darker());
        g.fillRect(column.x, column.y, column.width, column.height);
    }
    
    public void repaint(Graphics g) 
    {
        //Color The Background
        g.setColor(Color.cyan);
        g.fillRect(0, 0, WIDTH, HEIGHT);
        
        //Generate the ground
        g.setColor(Color.ORANGE);
        g.fillRect(0, HEIGHT - 120, WIDTH, 150);
        
        //Add grass to the ground
        g.setColor(Color.GREEN);
        g.fillRect(0, HEIGHT - 120, WIDTH, 20);
        
        //Represent the bird
        g.setColor(Color.RED);
        g.fillRect(bird.x, bird.y, bird.width, bird.height);
        
        //Iterate over the columns arraylist to generate the columns
        for(Rectangle column : columns)
        {
            paintColumn(g, column);
        }
        
        g.setColor(Color.WHITE);
        g.setFont(new Font("Arial", 1, 100));
        
        if(!started)
        {
            g.drawString("Click to Start", 100, HEIGHT / 2 - 50);
        }
        
        if(gameOver)
        { 
            g.drawString("Game Over!", 100, HEIGHT / 2 - 50);
        }
    }
    
    public static void main(String[] args) {
        flappyBird = new FlappyBird();
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        jump();
    }

    @Override
    public void mousePressed(MouseEvent e) {
    }

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }
}
