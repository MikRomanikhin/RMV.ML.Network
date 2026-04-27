module;

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

export module RMV.ML.MNIST.ConvUtils;

export namespace RMV::ML::MNIST::ConvUtils
{
   using MatrixXd = Eigen::MatrixXd;
   using Tensor4D = Eigen::Tensor<double, 4>;

   MatrixXd Im2Col( const Tensor4D& input, int FH, int FW, int stride, int pad )
   {
      int N = input.dimension( 0 );
      int C = input.dimension( 1 );
      int H = input.dimension( 2 );
      int W = input.dimension( 3 );

      int outH = ( H + 2 * pad - FH ) / stride + 1;
      int outW = ( W + 2 * pad - FW ) / stride + 1;

      MatrixXd col( N * outH * outW, C * FH * FW );

      int row_idx = 0;
      for( int n = 0; n < N; ++n )
      {
         for( int oh = 0; oh < outH; ++oh )
         {
            for( int ow = 0; ow < outW; ++ow )
            {
               int col_idx = 0;
               for( int c = 0; c < C; ++c )
               {
                  for( int fh = 0; fh < FH; ++fh )
                  {
                     for( int fw = 0; fw < FW; ++fw )
                     {
                        int ih = oh * stride - pad + fh;
                        int iw = ow * stride - pad + fw;

                        if( ih >= 0 && ih < H && iw >= 0 && iw < W )
                        {
                           col( row_idx, col_idx ) = input( n, c, ih, iw );
                        }
                        else
                        {
                           col( row_idx, col_idx ) = 0.0; // zero-padding
                        }
                        col_idx++;
                     }
                  }
               }
               row_idx++;
            }
         }
      }

      return col;
   }

   Tensor4D Col2Im( const MatrixXd& col, int N, int C, int H, int W, int FH, int FW, int stride, int pad )
   {
      int outH = ( H + 2 * pad - FH ) / stride + 1;
      int outW = ( W + 2 * pad - FW ) / stride + 1;

      Tensor4D img( N, C, H, W );
      img.setZero(); // Initialize all elements to 0 for accumulation

      int row_idx = 0;
      for( int n = 0; n < N; ++n )
      {
         for( int oh = 0; oh < outH; ++oh )
         {
            for( int ow = 0; ow < outW; ++ow )
            {
               int col_idx = 0;
               for( int c = 0; c < C; ++c )
               {
                  for( int fh = 0; fh < FH; ++fh )
                  {
                     for( int fw = 0; fw < FW; ++fw )
                     {
                        int ih = oh * stride - pad + fh;
                        int iw = ow * stride - pad + fw;

                        if( ih >= 0 && ih < H && iw >= 0 && iw < W )
                        {
                           // Accumulate the gradients back to the original image pixels
                           img( n, c, ih, iw ) += col( row_idx, col_idx );
                        }
                        col_idx++;
                     }
                  }
               }
               row_idx++;
            }
         }
      }

      return img;
   }
}