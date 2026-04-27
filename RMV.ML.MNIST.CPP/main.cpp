#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

// Import the modules we converted earlier
import RMV.ML.MNIST.AppSettings;
import RMV.ML.MNIST.Parser;
import RMV.ML.MNIST.DataSet;
import RMV.ML.MNIST.Controller;

using namespace RMV::ML::MNIST;

/// <summary>
/// Helper function to replicate C# File.ReadAllLines
/// </summary>
std::vector<std::string> ReadAllLines( const std::string& path )
{
   std::vector<std::string> lines;
   std::ifstream file( path );

   if( !file.is_open() )
      throw std::runtime_error( "Could not open file: " + path );

   std::string line;
   while( std::getline( file, line ) )
   {
      if( !line.empty() )
         lines.push_back( line );
   }

   return lines;
}

/// <summary>
/// Placeholder for ConfigManager.GetRoot<AppSettings>()
/// In a real C++ application, use a library like nlohmann/json to parse appSettings.json here.
/// </summary>
AppSettings LoadConfig()
{
   AppSettings settings;
   // Set appropriate default or hardcoded values here until JSON parsing is implemented  
   settings.TrainPath = "d://MNIST/mnist_train.csv"; // update with actual dataset path
   settings.TestPath = "d://MNIST/mnist_test.csv";   // update with actual dataset path     

   return settings;
}

int main()
{
   try
   {     
      AppSettings settings = LoadConfig();
            
      Parser parser( settings.Output );

      // Read dataset lines and parse them
      std::cout << "Reading datasets..." << std::endl;
      auto trainLines = ReadAllLines( settings.TrainPath );
      auto testLines = ReadAllLines( settings.TestPath );

      DataSet trainSet = parser.Run( trainLines );
      DataSet testSet = parser.Run( testLines );

      // Run the controller
      Controller controller( std::move( trainSet ), std::move( testSet ), settings );
      controller.RunML();
   }
   catch( const std::exception& e )
   {
      std::cerr << "An exception occurred: " << e.what() << std::endl;
      return 1;
   }

   return 0;
}