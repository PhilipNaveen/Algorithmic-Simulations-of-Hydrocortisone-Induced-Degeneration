#imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

class Bootstrap_SRSM:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def random_sampling_distribution(self, number_of_samples=10, sensitivity=10):
    #random samples list for x and y
        x_samples = []
        y_samples = []
        #for each sample
        for _ in range(number_of_samples):
            x_mean = 0
            y_mean = 0
            #sensitivity for means
            for _ in range(sensitivity):
                R = random.randint(0,74)
                x_mean += self.x[R][0]
                y_mean += self.y[R][0]
            x_samples.append([x_mean/sensitivity])
            y_samples.append([y_mean/sensitivity])
        return [x_samples, y_samples]

#declare class
class Regression:
    #declare method: fit and score regression
    @staticmethod
    def train_and_score(x, y):
        global x_test
        global x_train
        global y_test
        global y_train
        x_train, x_test, y_train, y_test = train_test_split(x,y)
        global model
        model = Sequential()
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.add(tf.keras.layers.Dense(5, activation="linear"))
        model.add(tf.keras.layers.Dense(7, activation="linear"))
        model.add(tf.keras.layers.Dense(7, activation="linear"))
        model.add(tf.keras.layers.Dense(7, activation="linear"))
        model.add(tf.keras.layers.Dense(5, activation="linear"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
        model.fit(x_train, y_train, epochs=5000)
        val_loss, val_mse, val_mae = model.evaluate(x_test, y_test)
    @staticmethod 
    def plot_generation():
       fig, axis = plt.subplots() 
       axis.set_title("Neural Network Fit to Bootstrapped Brix Data") 
       axis.set_ylabel("y-AXIS: Brix Score") 
       axis.set_xlabel("x-AXIS: Concentration of HC in Water Supply [%]") 
       #function to plot and show graph 
       axis.scatter(x_train, y_train)
       axis.plot(x_test, model.predict(x_test), color="red")
       plt.show()


#declare class:
class Emulator:
    def __init__(self, contaminator_size=0.0001, aquatic_volume=2.0000):
        self.contaminator_size = contaminator_size
        self.aquatic_volume = aquatic_volume
    #declare method to find concentration
    
    def concentration(self):
        #if elifs else
        self.concentration = None
        if (self.contaminator_size>=0 or self.aquatic_volume>=0):
            self.concentration = float(self.contaminator_size/(self.contaminator_size+self.aquatic_volume))
        else:
            pass
    #declare method for regression
    def estimation(self):
        return model.predict([[float(self.concentration)]])
        
if __name__ == "__main__":
    #declare data
    y_data = [
        [3.624],[3.339],[3.288],[3.452],[3.742],[3.254],[3.256],[3.291],[3.268],[3.403],[3.318],
        [3.685],[3.411],[3.699],[3.453],[3.136],[3.295],[2.946],[2.997],[3.364],[3.327],[3.612],
        [3.401],[3.194],[3.552],
        
        [3.213],[2.866],[3.039],[3.114],[3.214],[3.003],[2.495],[3.614],[2.841],[3.019],[2.943],
        [2.966],[3.204],[3.322],[2.878],[3.091],[3.168],[3.291],[2.921],[2.933],[3.096],[3.294],
        [2.897],[3.006],[3.294],
        
        [2.988],[2.446],[2.889],[2.213],[2.163],[2.771],[2.519],[2.316],[2.902],[2.663],[2.116],
        [2.399],[2.861],[2.809],[2.634],[2.455],[2.418],[2.604],[2.405],[2.333],[2.721],[2.679],
        [2.319],[2.906],[2.149]
    ]
    x_data = [
        [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
        [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
        [0],[0],[0],
        
        [0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],
        [0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],[0.025],
        [0.025],[0.025],[0.025],
        
        [0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],
        [0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],
        [0.05],[0.05],[0.05]
    ]
    #initialize the constructor from boostrap class
    bootstrap_object = Bootstrap_SRSM(x_data, y_data).random_sampling_distribution(number_of_samples=1000, sensitivity=500)
    #initialize the constructor from regression class
    Regression.train_and_score(bootstrap_object[0], bootstrap_object[1])
    Regression.plot_generation()
    #regular while loop
    user = int(input("Enter 1 for another cycle, enter 0 to quit: "))
    while (user==1):
        #takes all liquid size parameters in km^3 for units
        emulation_tool = Emulator(contaminator_size=0.0001, aquatic_volume=1.0000)
        emulation_tool.concentration()
        print(emulation_tool.estimation())
        user = int(input("Enter 1 for another cycle, enter 0 to quit: "))
        

    
    
    
    
    
    
    
    
    
    
    
    