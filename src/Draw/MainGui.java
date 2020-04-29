package Draw;

import javax.swing.*;
import java.awt.*;
import java.io.*;

public class MainGui extends JFrame {

    private final JPanel mainPanel;
    private DrawingPanel drawingPanel;

    private JButton clearButton;
    private JButton transformButton;
    private JButton trainNetworkButton;
    private JTextField trainingSetsAmount;
    private JTextArea outputTextArea;
    private JTextField expectedOutput;

    NeuralNetwork neuralNetwork = new NeuralNetwork();

    public static void main(String[] args) throws IOException {
        new MainGui();
    }

    public MainGui() throws IOException {
        super("Drawing letters using test.neural networks");
        mainPanel = new JPanel();
        mainPanel.setBackground(Color.LIGHT_GRAY);
        setContentPane(mainPanel);

        setLeftSide();
        setCenterArea();
        setOutputPanel();

        setOnClicks();

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setVisible(true);
        setSize(new Dimension(900, 500));
        setLocationRelativeTo(null);
        setResizable(false);
    }

    private void setLeftSide() {
        JPanel panel = new JPanel();
        panel.setBackground(Color.LIGHT_GRAY);
        panel.setPreferredSize(new Dimension(410, 440));
        int RESOLUTION = 20;
        drawingPanel = new DrawingPanel(400, 400, RESOLUTION);
        panel.add(drawingPanel);
        mainPanel.add(panel);
    }

    private void setCenterArea() {
        JPanel centerPanel = new JPanel();
        centerPanel.setLayout(new GridBagLayout());
        centerPanel.setPreferredSize(new Dimension(200, 400));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridwidth = GridBagConstraints.REMAINDER;
        gbc.anchor = GridBagConstraints.CENTER;

        trainNetworkButton = new JButton("Train X times:");
        trainingSetsAmount = new JFormattedTextField("5000");
        trainingSetsAmount.setMaximumSize(new Dimension(100, 30));
        trainingSetsAmount.setPreferredSize(new Dimension(100, 30));
        centerPanel.add(trainNetworkButton, gbc);
        centerPanel.add(trainingSetsAmount, gbc);

        centerPanel.add(Box.createVerticalStrut(50));

        expectedOutput = new JFormattedTextField("0");
        centerPanel.add(expectedOutput, gbc);

        centerPanel.add(Box.createVerticalStrut(50));

        transformButton = new JButton(">>");
        centerPanel.add(transformButton, gbc);

        centerPanel.add(Box.createVerticalStrut(50));

        clearButton = new JButton("Clear");
        clearButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        centerPanel.add(clearButton, gbc);

        mainPanel.add(centerPanel);
    }

    private void setOutputPanel() {
        JPanel outputPanel = new JPanel();
        outputPanel.setPreferredSize(new Dimension(200, 430));

        outputTextArea = new JTextArea();
        outputTextArea.setPreferredSize(new Dimension(200, 230));
        outputPanel.add(outputTextArea);

        mainPanel.add(outputPanel);
    }

    private void setOnClicks() {
        clearButton.addActionListener(e -> drawingPanel.clear());

        transformButton.addActionListener(e -> {
            float[] values = DrawingPanel.getSectionsValue();
            outputTextArea.setText("");
            neuralNetwork.train(1, 0.1f);
            neuralNetwork.forward(values);
            for (int i = 0; i < neuralNetwork.layers[neuralNetwork.layers.length - 1].neurons.length; i++) {
                outputTextArea.append(i + ": " + neuralNetwork.layers[neuralNetwork.layers.length - 1].neurons[i].value * 100);
                outputTextArea.append("\n");
            }
        });

        trainNetworkButton.addActionListener(e -> {
            int number = Integer.parseInt(trainingSetsAmount.getText());
            float[] values = values();
            float[] wee = wee();

            neuralNetwork.dataSet[expected()] = new DataSet(values, wee);
            neuralNetwork.train(number, 0.1f);

            try {
                saveToFile(Integer.toString(expected()), values);
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        });
    }

    private float[] values() {
        return DrawingPanel.getSectionsValue();
    }

    private int expected() {
        return Integer.parseInt(expectedOutput.getText());
    }

    private float[] wee() {
        int expected = expected();
        float[] wee = new float[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        if (expected == 0) {
            wee = new float[] {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        if (expected == 1) {
            wee = new float[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        if (expected == 2) {
            wee = new float[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        }
        if (expected == 3) {
            wee = new float[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
        }
        if (expected == 4) {
            wee = new float[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        }
        if (expected == 5) {
            wee = new float[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
        }
        if (expected == 6) {
            wee = new float[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
        }
        if (expected == 7) {
            wee = new float[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
        }
        if (expected == 8) {
            wee = new float[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
        }
        if (expected == 9) {
            wee = new float[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        }
        return wee;
    }

    private void saveToFile(String filename, float[] values) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (float value : values) {
            sb.append(value).append(" ");
        }
        String str = sb.toString();

        BufferedWriter writer = new BufferedWriter(new FileWriter("resources/" + filename + ".txt"));
        writer.write(str);
        writer.close();
    }
}
