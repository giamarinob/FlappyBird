package flappybird;
import javax.swing.JPanel;
import java.awt.Graphics;

/**
 *
 * @author User
 */
public class Renderer extends JPanel
{
    private static final long serialVersionUID = 1L;
    
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        
        FlappyBird.flappyBird.repaint(g);
    }
}
