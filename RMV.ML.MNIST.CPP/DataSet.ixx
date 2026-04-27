module;

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <tuple>
#include <Eigen/Core>

export module RMV.ML.MNIST.DataSet;

export namespace RMV::ML::MNIST
{
   using MatrixXd = Eigen::MatrixXd;

   /// <summary>
   /// Represents a collection of paired input and output data sets for machine learning or statistical analysis.
   /// </summary>
   class DataSet
   {
      public:
      std::vector<std::vector<double>> Source;
      std::vector<std::vector<double>> Target;

      DataSet( std::vector<std::vector<double>> source, std::vector<std::vector<double>> target )
         : Source( std::move( source ) ), Target( std::move( target ) )
      {}

      /// <summary>
      /// Builds random batch of source data arrays from the collection.
      /// </summary>
      DataSet GetRandomBatch( int batchSize ) const
      {
         std::vector<std::vector<double>> batchSource;
         std::vector<std::vector<double>> batchTarget;
         batchSource.reserve( batchSize );
         batchTarget.reserve( batchSize );

         std::vector<int> indices( Source.size() );
         std::iota( indices.begin(), indices.end(), 0 ); // Fill with 0, 1, 2...

         std::random_device rd;
         std::mt19937 gen( rd() );
         std::shuffle( indices.begin(), indices.end(), gen );

         for( int i = 0; i < batchSize; ++i )
         {
            batchSource.push_back( Source[ indices[ i ] ] );
            batchTarget.push_back( Target[ indices[ i ] ] );
         }

         return DataSet( std::move( batchSource ), std::move( batchTarget ) );
      }

      /// <summary>
      /// Selects a random batch of source data arrays from the collection and returns them as a tuple of matrices.
      /// </summary>
      std::tuple<MatrixXd, MatrixXd> GetRandomTrain( int batchSize ) const
      {
         int inputSize = Source.empty() ? 0 : static_cast< int >( Source[ 0 ].size() );
         int outputSize = Target.empty() ? 0 : static_cast< int >( Target[ 0 ].size() );

         std::vector<int> indices( Source.size() );
         std::iota( indices.begin(), indices.end(), 0 );

         std::random_device rd;
         std::mt19937 gen( rd() );
         std::shuffle( indices.begin(), indices.end(), gen );

         MatrixXd xBatch( batchSize, inputSize );
         MatrixXd tBatch( batchSize, outputSize );

         for( int row = 0; row < batchSize; ++row )
         {
            int idx = indices[ row ];
            for( int col = 0; col < inputSize; ++col )
            {
               xBatch( row, col ) = Source[ idx ][ col ];
            }
            for( int col = 0; col < outputSize; ++col )
            {
               tBatch( row, col ) = Target[ idx ][ col ];
            }
         }

         return { xBatch, tBatch };
      }

      /// <summary>
      /// Determines whether the maximum item index for the specified value equals the provided index.
      /// </summary>
      bool Match( int i, int index ) const
      {
         return GetMaxItemIndex( i ) == index;
      }

      private:
      /// <summary>
      /// Calculates index of the maximum value within the item array at the specified position.
      /// </summary>
      int GetMaxItemIndex( int i ) const
      {
         if( i < 0 || i >= Target.size() || Target[ i ].empty() )  return -1;

         auto it = std::max_element( Target[ i ].begin(), Target[ i ].end() );

         return static_cast<int>( std::distance( Target[ i ].begin(), it ) );
      }
   };
}