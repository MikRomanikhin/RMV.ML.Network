module;

#include <vector>
#include <string>
#include <sstream>
#include <optional>
#include <numeric>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <stdexcept>

export module RMV.ML.MNIST.Parser;

import RMV.ML.MNIST.DataSet;

export namespace RMV::ML::MNIST
{   
   /// <summary>
   /// Parses input data and maps output to binary vectors
   /// </summary>
   class Parser
   {
      std::optional<int> outputs;

      // Helper to split a CSV string into doubles
      static std::vector<double> ParseLineToDoubles( const std::string& line )
      {
         std::vector<double> result;
         std::stringstream ss( line );
         std::string token;
         while( std::getline( ss, token, ',' ) )
         {
            result.push_back( std::stod( token ) );
         }

         return result;
      }

      // Helper to split a CSV string into ints
      static std::vector<int> ParseLineToInts( const std::string& line )
      {
         std::vector<int> result;
         std::stringstream ss( line );
         std::string token;
         while( std::getline( ss, token, ',' ) )
         {
            result.push_back( std::stoi( token ) );
         }

         return result;
      }

      /// <summary>
      /// Normalizes input to [0, 1] range by dividing by a fixed scale (255 for image data).
      /// </summary>
      static std::vector<double> Normalize( const std::vector<double>& data )
      {
         std::vector<double> result( data.size() );
         std::transform( data.begin(), data.end(), result.begin(), []( double d ) { return d / 255.0; } );

         return result;
      }

      /// <summary>
      /// Normalizes input to have zero mean and unit variance
      /// </summary>
      static std::vector<double> NormalizeM( const std::vector<double>& data )
      {
         constexpr double DELTA = 1e-8; // small value to prevent division by zero
         double sum = std::accumulate( data.begin(), data.end(), 0.0 );
         double mean = sum / data.size();

         double variance = 0.0;
         for( double d : data )
         {
            variance += ( d - mean ) * ( d - mean );
         }

         variance /= data.size();

         double stdDev = std::sqrt( variance );

         std::vector<double> result( data.size() );
         std::transform( data.begin(), data.end(), result.begin(), [ mean, stdDev, DELTA ]( double d ) 
         {
            return ( d - mean ) / ( stdDev + DELTA );
         } );

         return result;
      }

      /// <summary>
      /// Binary mapping for output
      /// </summary>
      std::vector<double> BinaryVector( int value, double min = 0.0, double max = 1.0 ) const
      {
         if( !outputs.has_value() )
            throw std::runtime_error( "Outputs size must be defined to create a binary vector." );

         std::vector<double> result( outputs.value(), min );
         result[ value ] = max;
         return result;
      }

      public:
      explicit Parser( std::optional<int> outputs = std::nullopt ) : outputs( outputs ) {}

      /// <summary>
      /// Builds Train and Test data
      /// </summary>
      DataSet Run( const std::vector<std::string>& lines )
      {
         std::vector<std::vector<double>> input;
         std::vector<std::vector<double>> output;

         for( const std::string& line : lines )
         {
            std::vector<double> data = ParseLineToDoubles( line );
            if( data.empty() ) continue;

            // Extract all elements except the first one (label)
            std::vector<double> buffer( data.begin() + 1, data.end() );

            input.push_back( Normalize( buffer ) );
            output.push_back( BinaryVector( static_cast< int >( data[ 0 ] ) ) );
         }

         return DataSet( std::move( input ), std::move( output ) );
      }

      /// <summary>
      /// Parses an array of CSV-formatted strings and extracts image data as a list of integer arrays.
      /// </summary>
      static std::tuple<std::vector<std::vector<int>>, std::vector<int>> GetImages( const std::vector<std::string>& lines )
      {
         std::vector<std::vector<int>> images;
         std::vector<int> labels;

         for( const std::string& line : lines )
         {
            std::vector<int> data = ParseLineToInts( line );
            if( data.empty() ) continue;

            // Extract all elements except the first one (label)
            std::vector<int> buffer( data.begin() + 1, data.end() );

            images.push_back( std::move( buffer ) );
            labels.push_back( data[ 0 ] );
         }

         return { images, labels };
      }
   };
}