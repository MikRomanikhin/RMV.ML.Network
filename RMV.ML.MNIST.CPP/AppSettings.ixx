module;

#include <vector>
#include <string>

export module RMV.ML.MNIST.AppSettings;

export namespace RMV::ML::MNIST
{
   // These enums represent the types used in the configuration
   enum class ActivationType { Relu, Sigmoid };
   enum class LearningType { Default }; 
   enum class OptimizerType { SGD, Momentum, Adam, Nesterov, AdaGrad, RmsProp };

   /// <summary>
   /// Network configuration
   /// </summary>
   struct AppSettings
   {
      /// <summary>
      /// Input layer size
      /// </summary>
      int Input = 784;

      /// <summary>
      /// Dimensions of the input data e.g., channels, height, width for images
      /// </summary>
      //std::vector<int> InputDim;

      /// <summary>
      /// Filter parameters for convolutional layers e.g., number of filters, filter size, padding, stride
      /// </summary>
      //std::vector<int> Filters;

      double WeightInitStd = 0.01;

      /// <summary>
      /// Hidden layers sizes
      /// </summary>
      std::vector<int> Hidden {400,100};

      /// <summary>
      /// Output layer size
      /// </summary>
      int Output = 10;

      /// <summary>
      /// Activation function type
      /// </summary>
      ActivationType Activation = ActivationType::Relu;

      /// <summary>
      /// Learning type
      /// </summary>
      LearningType Learning = LearningType::Default;

      /// <summary>
      /// Learning rate
      /// </summary>	
      double Rate = 0.001;

      /// <summary>
      /// Momentum parameter
      /// </summary>
      double Momentum = 0.5;

      /// <summary>
      /// Optimizer type
      /// </summary>
      OptimizerType Optimizer = OptimizerType::Adam;

      /// <summary>
      /// Weight decay parameter
      /// </summary>
      double Decay = 0.0;

      /// <summary>
      /// Number of iterations
      /// </summary>
      int Iterations = 5000;

      /// <summary>
      /// Number of consecutive iterations without improvement.
      /// </summary>
      int Stagnation = 100;

      /// <summary>
      /// Batch size
      /// </summary>
      int Batch = 2000;

      /// <summary>
      /// Print interval
      /// </summary>
      int Print = 10;

      /// <summary>
      /// Validation interval
      /// </summary>
      int Epoch = 50;

      /// <summary>
      /// Train data file path
      /// </summary>
      std::string TrainPath;

      /// <summary>
      /// Test data file path
      /// </summary>
      std::string TestPath;

      /// <summary>
      /// Errors file path
      /// </summary>
      std::string ErrorPath;

      /// <summary>
      /// Indexes file path
      /// </summary>
      std::string IndexPath;
   };
}