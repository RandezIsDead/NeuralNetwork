package Draw;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class NeuralNetwork {

    public DataSet[] dataSet;
    public Layer[] layers;
    public ArrayList<float[]> v = new ArrayList<>();

    public NeuralNetwork() throws IOException {
        layers = new Layer[3];
        layers[0] = null;
        layers[1] = new Layer(400, 20);
        layers[2] = new Layer(20, 10);

        for (int i = 0; i < 10; i++) {
            v.add(loadDataSets("" + i));
//            values[i] = new float[25];
        }

        CreateDataSet();
    }

    float[] loadDataSets(String filename) throws IOException {
        float[] values = new float[400];

        BufferedReader reader = new BufferedReader(new FileReader("resources/" + filename + ".txt"));
        String currentLine = reader.readLine();
        reader.close();
        String[] tabOfFloatString = currentLine.split(" ");

        for(int i = 0; i < tabOfFloatString.length; i++){
            values[i] = Float.parseFloat(tabOfFloatString[i]);
        }
        return values;
    }

    public void CreateDataSet() {
        dataSet = new DataSet[10];

        dataSet[0] = new DataSet(v.get(0), new float[] {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        dataSet[1] = new DataSet(v.get(1), new float[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
        dataSet[2] = new DataSet(v.get(2), new float[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
        dataSet[3] = new DataSet(v.get(3), new float[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
        dataSet[4] = new DataSet(v.get(4), new float[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
        dataSet[5] = new DataSet(v.get(5), new float[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
        dataSet[6] = new DataSet(v.get(6), new float[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
        dataSet[7] = new DataSet(v.get(7), new float[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
        dataSet[8] = new DataSet(v.get(8), new float[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
        dataSet[9] = new DataSet(v.get(9), new float[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
    }

    public void forward(float[] inputs) {
        layers[0] = new Layer(inputs);
        for(int i = 1; i < layers.length; i++) {
            for(int j = 0; j < layers[i].neurons.length; j++) {
                float sum = 0;
                for(int k = 0; k < layers[i-1].neurons.length; k++) {
                    sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
                }
                sum += layers[i].neurons[j].bias;
                layers[i].neurons[j].value = Functions.Sigmoid(sum);
            }
        }
    }

    public void backward(float learning_rate, DataSet tData) {
        int number_layers = layers.length;
        int out_index = number_layers-1;
        for(int i = 0; i < layers[out_index].neurons.length; i++) {
            float output = layers[out_index].neurons[i].value;
            float target = tData.expectedOutput[i];
            float derivative = output-target;
            float delta = derivative*(output*(1-output));
            layers[out_index].neurons[i].gradient = delta;
            for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) {
                float previous_output = layers[out_index-1].neurons[j].value;
                float error = delta*previous_output;
                layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
            }
        }
        for(int i = out_index-1; i > 0; i--) {
            for(int j = 0; j < layers[i].neurons.length; j++) {
                float output = layers[i].neurons[j].value;
                float gradient_sum = gradientSum(j,i+1);
                float delta = (gradient_sum)*(output*(1-output));
                layers[i].neurons[j].gradient = delta;
                for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
                    float previous_output = layers[i-1].neurons[k].value;
                    float error = delta*previous_output;
                    layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
                }
            }
        }
        for (Layer layer : layers) {
            for (int j = 0; j < layer.neurons.length; j++) {
                layer.neurons[j].update_weight();
            }
        }
    }

    public float gradientSum(int c_index, int n_index) {
        float gradient_sum = 0;
        Layer current_layer = layers[n_index];
        for (int i = 0; i < current_layer.neurons.length; i++) {
            Neuron current_neuron = current_layer.neurons[i];
            gradient_sum += current_neuron.weights[c_index]*current_neuron.gradient;
        }
        return gradient_sum;
    }

    public void train(int iterations, float learning_rate) {
        for (int i = 0; i < iterations; i++) {
            for (DataSet dataSet : dataSet) {
                forward(dataSet.data);
                backward(learning_rate, dataSet);
            }
        }
    }
}

class Layer {

    Neuron[] neurons;

    //  Constructor for input Layer
    public Layer(float[] input) {
        this.neurons = new Neuron[input.length];
        for (int i = 0; i < input.length; i++) {
            neurons[i] = new Neuron(input[i]);
        }
    }

    //  Constructor for hidden and output Layers
    public Layer(int inNeurons, int insideNeurons) {
        this.neurons = new Neuron[insideNeurons];

        for (int i = 0; i < insideNeurons; i++) {
            float[] weights = new float[inNeurons];
            for (int j = 0; j < inNeurons; j++) {
                weights[j] = Functions.RandomFloat(-1, 1);
            }
            neurons[i] = new Neuron(Functions.RandomFloat(0,1), weights);
        }
    }
}

class Neuron {

    public float value;
    float gradient;
    float[] weights;
    float bias;
    float[] cache_weights;

    public Neuron(float[] weights, float bias){
        this.weights = weights;
        this.bias = bias;
        this.cache_weights = this.weights;
        this.gradient = 0;
    }

    public Neuron(float value, float[] weights){
        this.weights = weights;
        this.value = value;
        this.cache_weights = this.weights;
        this.gradient = 0;
    }

    public Neuron(float value){
        this.weights = null;
        this.bias = -1;
        this.gradient = -1;
        this.value = value;
    }

    public void update_weight() {
        this.weights = this.cache_weights;
    }
}

class DataSet {

    public float[] data;
    float[] expectedOutput;

    public DataSet(float[] data) {
        this.data = data;
    }

    public DataSet(float[] data, float[] expectedOutput) {
        this.data = data;
        this.expectedOutput = expectedOutput;
    }
}

class Functions {

    public static float RandomFloat(float min, float max) {
        float a = (float) Math.random();
        float num = min + (float) Math.random() * (max - min);
        if(a < 0.5)
            return num;
        else
            return -num;
    }

    public static float Sigmoid(float x) {
        return (float) (1/(1+Math.pow(Math.E, -x)));
    }

    public static float SigmoidDerivative(float x) {
        return Sigmoid(x)*(1-Sigmoid(x));
    }

    public static float squaredError(float output,float target) {
        return (float) (0.5*Math.pow(2,(target-output)));
    }

    public static float sumSquaredError(float[] outputs,float[] targets) {
        float sum = 0;
        for(int i=0;i<outputs.length;i++) {
            sum += squaredError(outputs[i],targets[i]);
        }
        return sum;
    }
}
