module;

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <tuple>
#include <stdexcept>
#include <Eigen/Core>

export module RMV.ML.MNIST.MultiLayerNet;

import RMV.ML.MNIST.AppSettings;
import RMV.ML.MNIST.Optimizer;
import RMV.ML.MNIST.Layers;


export namespace RMV::ML::MNIST
{
   using MatrixXd = Eigen::MatrixXd;
   using VectorXd = Eigen::VectorXd;

   // Supporting structures translated from AppSettings and IOptimizer
   /*export enum class ActivationType { Relu, Sigmoid };
   export enum class OptimizerType { SGD, Momentum, Adam, Nesterov, AdaGrad, RmsProp };*/

  /* export struct AppSettings
   {
      int Input;
      int Output;
      std::vector<int> Hidden;
      double Decay;
      ActivationType Activation;
      OptimizerType Optimizer;
   };*/

  /* export class IOptimizer
   {
      public:
      virtual ~IOptimizer() = default;
      virtual void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                           const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) = 0;
   };*/

   // Creates Optimizer based on configuration settings
   std::unique_ptr<IOptimizer> BuildOptimizer( const AppSettings& settings )
   {
      switch( settings.Optimizer )
      {
         case OptimizerType::SGD:
            return std::make_unique<SGD>( settings.Rate );

         case OptimizerType::Momentum:
            return std::make_unique<Momentum>( settings.Rate, settings.Momentum );

         case OptimizerType::Nesterov:
            return std::make_unique<Nesterov>( settings.Rate, settings.Momentum );

         case OptimizerType::AdaGrad:
            return std::make_unique<AdaGrad>( settings.Rate );

         case OptimizerType::RmsProp:
            // RmsProp takes rate and an optional decay. We use defaults for decay here if not in settings.
            return std::make_unique<RmsProp>( settings.Rate );

         case OptimizerType::Adam:
            // Adam takes rate, beta1, and beta2. We pass rate and let others use defaults.
            return std::make_unique<Adam>( settings.Rate );

         default:
            throw std::runtime_error( "Unsupported optimizer specified in settings." );
      }
   }


   /// <summary>
   /// Fully connected deep neural network with arbitrary hidden layers.
   /// </summary>
   export class MultiLayerNet
   {
      int inputSize;
      int outputSize;
      std::vector<int> HiddenSizeList;
      int hiddenLayerNum;
      double weightDecayLambda;

      std::vector<MatrixXd> Weights;
      std::vector<VectorXd> Biases;

      std::vector<std::shared_ptr<ILayer>> Layers;
      std::vector<std::shared_ptr<Affine>> AffineLayers;

      SoftmaxWithLoss LastLayer;
      std::unique_ptr<IOptimizer> optimizer;

      void InitWeight( ActivationType activation )
      {
         std::vector<int> allSizeList = { inputSize };
         allSizeList.insert( allSizeList.end(), HiddenSizeList.begin(), HiddenSizeList.end() );
         allSizeList.push_back( outputSize );

         std::random_device rd;
         std::mt19937 gen( rd() );

         for( size_t i = 0; i < allSizeList.size() - 1; i++ )
         {
            double scale = 0.01;
            if( activation == ActivationType::Relu )
               scale = std::sqrt( 2.0 / allSizeList[ i ] ); // ReLU (He initialization)
            else if( activation == ActivationType::Sigmoid )
               scale = std::sqrt( 1.0 / allSizeList[ i ] ); // Sigmoid (Xavier initialization)

            std::normal_distribution<double> d( 0.0, scale );

            // Populate random normally distributed values
            Weights[ i ] = MatrixXd::NullaryExpr( allSizeList[ i ], allSizeList[ i + 1 ], [ & ]() { return d( gen ); } );
            Biases[ i ] = VectorXd::Zero( allSizeList[ i + 1 ] );
         }
      }

      public:
      MultiLayerNet( const AppSettings& settings )
      {
         inputSize = settings.Input;
         HiddenSizeList = settings.Hidden;
         outputSize = settings.Output;
         hiddenLayerNum = static_cast< int >( settings.Hidden.size() );
         weightDecayLambda = settings.Decay;

         int totalLayers = hiddenLayerNum + 1;
         Weights.resize( totalLayers );
         Biases.resize( totalLayers );
         AffineLayers.resize( totalLayers );

         optimizer = BuildOptimizer( settings );
         InitWeight( settings.Activation );

         for( int i = 0; i < hiddenLayerNum; i++ )
         {
            auto affine = std::make_shared<Affine>( Weights[ i ], Biases[ i ] );
            AffineLayers[ i ] = affine;
            Layers.push_back( affine );

            if( settings.Activation == ActivationType::Relu )
               Layers.push_back( std::make_shared<Relu>() );
            else if( settings.Activation == ActivationType::Sigmoid )
               Layers.push_back( std::make_shared<Sigmoid>() );
         }

         int lastIdx = hiddenLayerNum;
         auto lastAffine = std::make_shared<Affine>( Weights[ lastIdx ], Biases[ lastIdx ] );
         AffineLayers[ lastIdx ] = lastAffine;
         Layers.push_back( lastAffine );
      }

      MatrixXd Predict( MatrixXd x )
      {
         for( auto& layer : Layers )
         {
            x = layer->Forward( x );
         }
         return x;
      }

      double Loss( const MatrixXd& x, const MatrixXd& t )
      {
         MatrixXd y = Predict( x );
         double weightDecay = 0.0;

         for( const auto& w : Weights )
         {
            weightDecay += 0.5 * weightDecayLambda * w.squaredNorm();
         }

         return LastLayer.Forward( y, t ) + weightDecay;
      }

      std::tuple<double, std::vector<int>, std::vector<int>> Accuracy( const MatrixXd& x, const MatrixXd& t )
      {
         MatrixXd y = Predict( x );
         int batchSize = static_cast< int >( y.rows() );
         int correct = 0;
         std::vector<int> errors;
         std::vector<int> indexes;

         for( int i = 0; i < batchSize; i++ )
         {
            int yPred, tIndex;
            y.row( i ).maxCoeff( &yPred ); // Index of max coefficient

            if( t.cols() == 1 )
               tIndex = static_cast< int >( t( i, 0 ) );
            else
               t.row( i ).maxCoeff( &tIndex );

            if( yPred == tIndex )
            {
               correct++;
            }
            else
            {
               errors.push_back( yPred );
               indexes.push_back( i );
            }
         }

         double acc = static_cast< double >( correct ) / batchSize;
         return { acc, errors, indexes };
      }

      std::pair<std::vector<MatrixXd>, std::vector<VectorXd>> Gradient( const MatrixXd& x, const MatrixXd& t )
      {         
         Loss( x, t ); // Forward pass
                  
         MatrixXd dout = LastLayer.Backward( 1.0 );  // Backward pass

         for( auto it = Layers.rbegin(); it != Layers.rend(); ++it )
         {
            dout = ( *it )->Backward( dout );
         }

         std::vector<MatrixXd> dW( AffineLayers.size() );
         std::vector<VectorXd> dB( AffineLayers.size() );

         for( size_t i = 0; i < AffineLayers.size(); i++ )
         {
            dW[ i ] = AffineLayers[ i ]->dW.value() + ( weightDecayLambda * Weights[ i ] );
            dB[ i ] = AffineLayers[ i ]->dB.value();
         }

         return { dW, dB };
      }

      void Update( const MatrixXd& x, const MatrixXd& t )
      {
         auto [dW, dB] = Gradient( x, t );
         optimizer->Update( Weights, Biases, dW, dB );
      }
   };
}