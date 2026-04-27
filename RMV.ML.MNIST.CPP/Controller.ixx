module;

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <tuple>
#include <Eigen/Core>

export module RMV.ML.MNIST.Controller;

import RMV.ML.MNIST.AppSettings;
import RMV.ML.MNIST.DataSet;
import RMV.ML.MNIST.MultiLayerNet; 

export namespace RMV::ML::MNIST
{
   using MatrixXd = Eigen::MatrixXd;   

   /// <summary>
   /// Neural network training using various optimization algorithms
   /// </summary>
   class Controller
   {
      DataSet trainSet;
      DataSet testSet;
      AppSettings settings;

      static constexpr int LINE_LENGTH = 40;

      // Helper to convert std::vector<std::vector<double>> to Eigen::MatrixXd
      MatrixXd VectorToMatrix( const std::vector<std::vector<double>>& vec )
      {
         int rows = static_cast< int >( vec.size() );
         int cols = rows > 0 ? static_cast< int >( vec[ 0 ].size() ) : 0;
         MatrixXd mat( rows, cols );
         for( int i = 0; i < rows; ++i )
            for( int j = 0; j < cols; ++j )
               mat( i, j ) = vec[ i ][ j ];
         return mat;
      }

      // Helper to format chrono duration to hh:mm:ss string
      std::string FormatTime( std::chrono::seconds totalSeconds )
      {
         auto h = std::chrono::duration_cast< std::chrono::hours >( totalSeconds );
         totalSeconds -= h;
         auto m = std::chrono::duration_cast< std::chrono::minutes >( totalSeconds );
         totalSeconds -= m;

         char buf[ 32 ];
         snprintf( buf, sizeof( buf ), "%02d:%02d:%02d", static_cast< int >( h.count() ), static_cast< int >( m.count() ), static_cast< int >( totalSeconds.count() ) );
         return std::string( buf );
      }

      // Helper to write generic comma-separated lists to file (to match C# string.Join())
      void SaveListToFile( const std::string& path, const std::vector<int>& list )
      {
         std::ofstream out( path );
         if( !out.is_open() ) return;
         for( size_t i = 0; i < list.size(); ++i )
         {
            out << list[ i ];
            if( i + 1 < list.size() ) out << ",";
         }
      }

      public:
      Controller( DataSet train, DataSet test, AppSettings appSettings )
         : trainSet( std::move( train ) ), testSet( std::move( test ) ), settings( std::move( appSettings ) )
      {}

      /// <summary>
      /// Trains and evaluates a neural network using the configured training and validation data sets.
      /// </summary>
      void RunML()
      {
         using clock = std::chrono::high_resolution_clock;
         auto startTime = clock::now();

         double maxAccuracy = std::numeric_limits<double>::lowest();

         MatrixXd valX = VectorToMatrix( testSet.Source );
         MatrixXd valT = VectorToMatrix( testSet.Target );

         auto elapsedInit = std::chrono::duration<double>( clock::now() - startTime ).count();
         std::cout << "Loaded data sets. Time:" << std::fixed << std::setprecision( 2 ) << elapsedInit << " sec\n";

         MultiLayerNet network( settings );

         int stagnation = 0;
         int bestIter = 0;

         for( int i = 0; i < settings.Iterations; i++ )
         {
            auto [xBatch, tBatch] = trainSet.GetRandomTrain( settings.Batch );

            network.Update( xBatch, tBatch );   // update the network weights based on the current batch

            double loss = network.Loss( xBatch, tBatch ); // calculate the loss for the current batch

            auto [trainAcc, trainErrors, trainIndexes] = network.Accuracy( xBatch, tBatch );  // training accuracy for the current batch

            auto currentElapsed = std::chrono::duration_cast< std::chrono::seconds >( clock::now() - startTime );

            if( i % settings.Print == 0 )
            {
               std::cout << "iter:" << i << " loss:" << std::fixed << std::setprecision( 4 ) << loss
                  << "  accuracy:" << trainAcc << "  Time=" << FormatTime( currentElapsed ) << "\n";
            }

            if( i % settings.Epoch == 0 ) // validation testing at configured intervals
            {
               auto [testAcc, errors, indexes] = network.Accuracy( valX, valT ); // validation accuracy, errors, and indexes

               std::cout << "iter:" << i << "  validation:" << std::fixed << std::setprecision( 4 ) << testAcc
                  << "  Time=" << FormatTime( currentElapsed ) << "\n";
               std::cout << std::string( LINE_LENGTH, '-' ) << "\n";

               if( testAcc > maxAccuracy ) // if the accuracy is better than the best so far, save the errors to a file
               {
                  maxAccuracy = testAcc;
                  stagnation = 0;
                  bestIter = i;
                  // ErrorPath and IndexPath need to be added to your AppSettings struct
                  SaveListToFile( settings.ErrorPath, errors );
                  SaveListToFile( settings.IndexPath, indexes );
                  continue;
               }

               if( ++stagnation > settings.Stagnation ) // no improvement in validation accuracy - stop training
               {
                  auto stopElapsed = std::chrono::duration_cast< std::chrono::seconds >( clock::now() - startTime );
                  std::cout << "Stopping at " << i << " due to stagnation. Accuracy: "
                     << std::fixed << std::setprecision( 4 ) << maxAccuracy
                     << " at iteration " << bestIter << ". Time=" << FormatTime( stopElapsed ) << "\n";
                  break;
               }
            }
         }
      }
   };
}