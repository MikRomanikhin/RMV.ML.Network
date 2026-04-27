module;

#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <optional>
#include <algorithm>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

export module RMV.ML.MNIST.Layers;

import RMV.ML.MNIST.ConvUtils;

// Aliasing Eigen types for easier usage throughout the layers
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;


export namespace RMV::ML::MNIST
{

   /// <summary>
   /// Common interface for layers supporting batch (matrix) forward and backward passes.
   /// </summary>
   class ILayer
   {
      public:
      virtual ~ILayer() = default;
      virtual MatrixXd Forward( const MatrixXd& x ) = 0;
      virtual MatrixXd Backward( const MatrixXd& dout ) = 0;
   };

   /// <summary>
   /// Rectified Linear Unit (ReLU) activation function layer
   /// </summary>
   class Relu : public ILayer
   {     
      std::optional<MatrixXd> matrixMask;

      public:
      MatrixXd Forward( const MatrixXd& x ) override
      {
         matrixMask = x.unaryExpr( []( double val ) { return val > 0 ? 1.0 : 0.0; } );

         return x.cwiseProduct( *matrixMask );
      }

      MatrixXd Backward( const MatrixXd& dout ) override
      {
         if( !matrixMask.has_value() )
            throw std::logic_error( "Forward must be called before Backward." );

         return dout.cwiseProduct( *matrixMask );
      }
   };

   /// <summary>
   /// Sigmoid activation function layer
   /// </summary>
   class Sigmoid : public ILayer
   {
      std::optional<MatrixXd> outMatrix;

      static double SigmoidFunc( double x )
      {
         return 1.0 / ( 1.0 + std::exp( -x ) );
      }

      public:
      MatrixXd Forward( const MatrixXd& x ) override
      {
         outMatrix = x.unaryExpr( &SigmoidFunc );
         return *outMatrix;
      }

      MatrixXd Backward( const MatrixXd& dout ) override
      {
         if( !outMatrix.has_value() )
            throw std::logic_error( "Forward must be called before Backward." );

         auto yOneMinusY = outMatrix->unaryExpr( []( double y ) { return y * ( 1.0 - y ); } );
         return dout.cwiseProduct( yOneMinusY );
      }
   };

   /// <summary>
   /// Affine (Fully Connected) layer: computes output = input * W + b
   /// </summary>
   class Affine : public ILayer
   {
      MatrixXd* W;
      VectorXd* B;
      std::optional<MatrixXd> x;

      public:
      std::optional<MatrixXd> dW;
      std::optional<VectorXd> dB;

      Affine( MatrixXd& w, VectorXd& b ) : W( &w ), B( &b ) {}

      /// <summary>
      /// Performs the forward pass of a linear (fully connected) layer
      /// </summary>
      /// <param name="input">The input matrix where each row represents a sample and each column represents a feature.</param>
      /// <returns>The output matrix after applying the linear transformation (input * weights + bias).</returns>
      MatrixXd Forward( const MatrixXd& input ) override
      {
         x = input;
         MatrixXd outMatrix = ( *x ) * ( *W );
                  
         outMatrix.rowwise() += B->transpose(); // Efficient broadcast addition of Bias vector to every row

         return outMatrix;
      }

      /// <summary>
      /// Column-wise sum returns a row vector, transpose to convert it back to a standard column VectorXd
      /// </summary>
      /// <param name="dout">Upstream gradient</param>
      /// <returns>Gradient with respect to the input</returns>
      MatrixXd Backward( const MatrixXd& dout ) override
      {
         if( !x.has_value() )         
            throw std::logic_error( "Forward must be called before Backward." );
         
         MatrixXd dx = dout * W->transpose();
         
         dW = x->transpose() * dout;
         
         dB = dout.colwise().sum().transpose();
         
         return dx;
      }
   };


   /// <summary>
   /// Softmax activation combined with Cross-Entropy Loss layer.
   /// </summary>
   class SoftmaxWithLoss
   {
      std::optional<MatrixXd> Y;
      std::optional<MatrixXd> T;

      static MatrixXd Softmax( const MatrixXd& x )
      {
         // Subtract max per row for numerical stability
         VectorXd rowMax = x.rowwise().maxCoeff();
         MatrixXd expRow = ( x.colwise() - rowMax ).array().exp();
         VectorXd rowSum = expRow.rowwise().sum();
         return expRow.array().colwise() / rowSum.array();
      }

      static double CrossEntropyError( const MatrixXd& y, const MatrixXd& t )
      {
         int batchSize = y.rows();
         const double delta = 1e-7;

         if( t.cols() == y.cols() ) // One-hot encoded
         {
            return -( t.array() * ( y.array() + delta ).log() ).sum() / batchSize;
         }
         else // Label indices
         {
            double loss = 0;
            for( int i = 0; i < batchSize; i++ )
            {
               int label = static_cast< int >( t( i, 0 ) );
               loss -= std::log( y( i, label ) + delta );
            }
            return loss / batchSize;
         }
      }

      public:
      double Forward( const MatrixXd& x, const MatrixXd& t )
      {
         T = t;
         Y = Softmax( x );
         return CrossEntropyError( *Y, *T );
      }

      MatrixXd Backward( double dout = 1.0 )
      {
         if( !Y.has_value() || !T.has_value() )
            throw std::logic_error( "Forward must be called before Backward." );

         int batchSize = T->rows();
         int classes = Y->cols();
         MatrixXd dx = MatrixXd::Zero( batchSize, classes );

         if( T->cols() == Y->cols() )
         {
            dx = ( *Y - *T ) / batchSize;
         }
         else
         {
            dx = *Y;
            for( int i = 0; i < batchSize; i++ )
            {
               int label = static_cast< int >( ( *T )( i, 0 ) );
               dx( i, label ) -= 1.0;
            }
            dx /= batchSize;
         }

         if( dout != 1.0 ) dx *= dout;
         return dx;
      }
   };

   /// <summary>
   /// Dropout layer
   /// </summary>
   class Dropout : public ILayer
   {
      double ratio;
      std::optional<MatrixXd> mask;

      public:
      bool TrainFlag = true;

      explicit Dropout( double ratio = 0.5 ) : ratio( ratio ) {}

      MatrixXd Forward( const MatrixXd& x ) override
      {
         if( TrainFlag )
         {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::uniform_real_distribution<> dis( 0.0, 1.0 );

            // Capture mutable distributions sequentially during matrix iteration
            mask = x.unaryExpr( [ & ]( double ) { return dis( gen ) > ratio ? 1.0 : 0.0; } );
            return x.cwiseProduct( *mask );
         }

         return x * ( 1.0 - ratio );
      }

      MatrixXd Backward( const MatrixXd& dout ) override
      {
         if( !mask.has_value() )         
            throw std::logic_error( "Forward must be called before Backward." );         

         return dout.cwiseProduct( *mask );
      }
   };


   /// <summary>
  /// Convolution layer
  /// </summary>
   class Convolution
   {
      Tensor4D W;
      VectorXd B;
      int stride;
      int pad;

      std::optional<Tensor4D> x;
      std::optional<MatrixXd> col;
      std::optional<MatrixXd> colW;

      public:
      std::optional<Tensor4D> dW;
      std::optional<VectorXd> dB;

      Convolution( Tensor4D w, VectorXd b, int stride = 1, int pad = 0 )
         : W( std::move( w ) ), B( std::move( b ) ), stride( stride ), pad( pad )
      {}

      Tensor4D Forward( const Tensor4D& input )
      {
         int FN = W.dimension( 0 );
         int C = W.dimension( 1 );
         int FH = W.dimension( 2 );
         int FW = W.dimension( 3 );

         int N = input.dimension( 0 );
         int H = input.dimension( 2 );
         int inW = input.dimension( 3 );

         int outH = 1 + ( H + 2 * pad - FH ) / stride;
         int outW = 1 + ( inW + 2 * pad - FW ) / stride;

         col = ConvUtils::Im2Col( input, FH, FW, stride, pad );

         colW = MatrixXd( C * FH * FW, FN );
         for( int fn = 0; fn < FN; fn++ )
         {
            int idx = 0;
            for( int c = 0; c < C; c++ )
               for( int fh = 0; fh < FH; fh++ )
                  for( int fw = 0; fw < FW; fw++ )
                     ( *colW )( idx++, fn ) = W( fn, c, fh, fw );
         }

         MatrixXd outMatrix = ( *col ) * ( *colW );
         outMatrix.rowwise() += B.transpose(); // Add bias to each row

         Tensor4D result( N, FN, outH, outW );
         int r = 0;
         for( int n = 0; n < N; n++ )
            for( int oh = 0; oh < outH; oh++ )
               for( int ow = 0; ow < outW; ow++ )
               {
                  for( int fn = 0; fn < FN; fn++ ) result( n, fn, oh, ow ) = outMatrix( r, fn );
                  r++;
               }

         x = input;
         return result;
      }

      Tensor4D Backward( const Tensor4D& dout )
      {
         if( !x.has_value() || !col.has_value() || !colW.has_value() )
            throw std::logic_error( "Forward must be called before Backward." );

         int FN = W.dimension( 0 );
         int C = W.dimension( 1 );
         int FH = W.dimension( 2 );
         int FW = W.dimension( 3 );

         int N = dout.dimension( 0 );
         int outH = dout.dimension( 2 );
         int outW = dout.dimension( 3 );

         MatrixXd doutMatrix( N * outH * outW, FN );
         int r = 0;
         for( int n = 0; n < N; n++ )
            for( int oh = 0; oh < outH; oh++ )
               for( int ow = 0; ow < outW; ow++ )
               {
                  for( int fn = 0; fn < FN; fn++ ) doutMatrix( r, fn ) = dout( n, fn, oh, ow );
                  r++;
               }

         dB = doutMatrix.colwise().sum().transpose();
         MatrixXd dWMatrix = col->transpose() * doutMatrix;

         dW = Tensor4D( FN, C, FH, FW );
         for( int fn = 0; fn < FN; fn++ )
         {
            int idx = 0;
            for( int c = 0; c < C; c++ )
               for( int fh = 0; fh < FH; fh++ )
                  for( int fw = 0; fw < FW; fw++ )
                     ( *dW )( fn, c, fh, fw ) = dWMatrix( idx++, fn );
         }

         MatrixXd dcol = doutMatrix * colW->transpose();

         return ConvUtils::Col2Im( dcol, x->dimension( 0 ), x->dimension( 1 ), x->dimension( 2 ), x->dimension( 3 ), FH, FW, stride, pad );
      }
   };

   /// <summary>
   /// Max pooling layer
   /// </summary>
   class Pooling
   {
      int poolH, poolW, stride, pad;
      std::optional<Tensor4D> x;
      std::optional<std::vector<int>> argMax;

      public:
      Pooling( int poolH, int poolW, int stride = 2, int pad = 0 )
         : poolH( poolH ), poolW( poolW ), stride( stride ), pad( pad )
      {}

      Tensor4D Forward( const Tensor4D& input )
      {
         int N = input.dimension( 0 );
         int C = input.dimension( 1 );
         int H = input.dimension( 2 );
         int W = input.dimension( 3 );

         int outH = 1 + ( H - poolH ) / stride;
         int outW = 1 + ( W - poolW ) / stride;

         MatrixXd col = ConvUtils::Im2Col( input, poolH, poolW, stride, pad );

         int poolSize = poolH * poolW;
         int totalPatches = N * C * outH * outW;
         MatrixXd colReshaped( totalPatches, poolSize );

         int srcRow = 0;
         for( int n = 0; n < N; n++ )
         {
            for( int oh = 0; oh < outH; oh++ )
               for( int ow = 0; ow < outW; ow++ )
               {
                  for( int c = 0; c < C; c++ )
                  {
                     int dstRow = ( ( n * C + c ) * outH + oh ) * outW + ow;
                     for( int p = 0; p < poolSize; p++ )
                        colReshaped( dstRow, p ) = col( srcRow, c * poolSize + p );
                  }
                  srcRow++;
               }
         }

         argMax = std::vector<int>( totalPatches );
         Tensor4D result( N, C, outH, outW );

         for( int i = 0; i < totalPatches; i++ )
         {
            int maxIdx;
            double max = colReshaped.row( i ).maxCoeff( &maxIdx );
            ( *argMax )[ i ] = maxIdx;

            int ow2 = i % outW;
            int oh2 = ( i / outW ) % outH;
            int c2 = ( i / ( outW * outH ) ) % C;
            int n2 = i / ( outW * outH * C );
            result( n2, c2, oh2, ow2 ) = max;
         }

         x = input;
         return result;
      }

      Tensor4D Backward( const Tensor4D& dout )
      {
         if( !x.has_value() || !argMax.has_value() )
            throw std::logic_error( "Forward must be called before Backward." );

         int N = dout.dimension( 0 );
         int C = dout.dimension( 1 );
         int outH = dout.dimension( 2 );
         int outW = dout.dimension( 3 );

         int poolSize = poolH * poolW;
         int totalPatches = N * C * outH * outW;

         MatrixXd dmax = MatrixXd::Zero( totalPatches, poolSize );
         for( int i = 0; i < totalPatches; i++ )
         {
            int ow2 = i % outW;
            int oh2 = ( i / outW ) % outH;
            int c2 = ( i / ( outW * outH ) ) % C;
            int n2 = i / ( outW * outH * C );
            dmax( i, ( *argMax )[ i ] ) = dout( n2, c2, oh2, ow2 );
         }

         MatrixXd dcol( N * outH * outW, C * poolSize );
         int dstRow = 0;
         for( int n = 0; n < N; n++ )
         {
            for( int oh = 0; oh < outH; oh++ )
               for( int ow = 0; ow < outW; ow++ )
               {
                  for( int c = 0; c < C; c++ )
                  {
                     int srcRow = ( ( n * C + c ) * outH + oh ) * outW + ow;
                     for( int p = 0; p < poolSize; p++ )
                        dcol( dstRow, c * poolSize + p ) = dmax( srcRow, p );
                  }
                  dstRow++;
               }
         }

         return ConvUtils::Col2Im( dcol, N, C, x->dimension( 2 ), x->dimension( 3 ), poolH, poolW, stride, pad );
      }
   };
}