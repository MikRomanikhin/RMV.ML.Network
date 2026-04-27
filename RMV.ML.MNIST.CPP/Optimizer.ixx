module;

#include <vector>
#include <cmath>
#include <Eigen/Core>

export module RMV.ML.MNIST.Optimizer;

export namespace RMV::ML::MNIST
{
   using MatrixXd = Eigen::MatrixXd;
   using VectorXd = Eigen::VectorXd;

   /// <summary>
   /// Interface for optimization algorithms used in training neural networks.
   /// </summary>
   class IOptimizer
   {
      public:
      virtual ~IOptimizer() = default;
      virtual void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                           const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) = 0;
   };

   /// <summary>
   /// Base class for implementing optimization algorithms that update weights and biases
   /// </summary>
   class BaseOptimizer : public IOptimizer
   {
      protected:
      std::vector<MatrixXd> vW;
      std::vector<VectorXd> vB;

      void Initialize( const std::vector<MatrixXd>& weights, const std::vector<VectorXd>& biases )
      {
         if( vW.empty() )
         {
            vW.resize( weights.size() );
            vB.resize( biases.size() );

            for( size_t i = 0; i < weights.size(); i++ )
            {
               vW[ i ] = MatrixXd::Zero( weights[ i ].rows(), weights[ i ].cols() );
               vB[ i ] = VectorXd::Zero( biases[ i ].size() );
            }
         }
      }
   };

   /// <summary>
   /// Stochastic Gradient Descent
   /// </summary>
   class SGD : public IOptimizer
   {
      double rate;

      public:
      explicit SGD( double rate = 0.01 ) : rate( rate ) {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         for( size_t i = 0; i < weights.size(); i++ )
         {
            weights[ i ] -= rate * dW[ i ];
            biases[ i ] -= rate * dB[ i ];
         }
      }
   };

   /// <summary>
   /// Momentum SGD
   /// </summary>
   class Momentum : public BaseOptimizer
   {
      double rate;
      double momentum;

      public:
      explicit Momentum( double rate = 0.01, double momentum = 0.9 )
         : rate( rate ), momentum( momentum )
      {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         Initialize( weights, biases );

         for( size_t i = 0; i < weights.size(); i++ )
         {
            vW[ i ] = momentum * vW[ i ] - rate * dW[ i ];
            weights[ i ] += vW[ i ];

            vB[ i ] = momentum * vB[ i ] - rate * dB[ i ];
            biases[ i ] += vB[ i ];
         }
      }
   };

   /// <summary>
   /// Nesterov's Accelerated Gradient
   /// </summary>
   class Nesterov : public BaseOptimizer
   {
      double rate;
      double momentum;

      public:
      explicit Nesterov( double rate = 0.01, double momentum = 0.9 )
         : rate( rate ), momentum( momentum )
      {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         Initialize( weights, biases );

         for( size_t i = 0; i < weights.size(); i++ )
         {
            MatrixXd vWPrev = vW[ i ];
            vW[ i ] = momentum * vW[ i ] - rate * dW[ i ];
            weights[ i ] += -momentum * vWPrev + ( 1.0 + momentum ) * vW[ i ];

            VectorXd vBPrev = vB[ i ];
            vB[ i ] = momentum * vB[ i ] - rate * dB[ i ];
            biases[ i ] += -momentum * vBPrev + ( 1.0 + momentum ) * vB[ i ];
         }
      }
   };

   /// <summary>
   /// AdaGrad
   /// </summary>
   class AdaGrad : public BaseOptimizer
   {
      double rate;

      public:
      explicit AdaGrad( double rate = 0.01 ) : rate( rate ) {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         Initialize( weights, biases );

         for( size_t i = 0; i < weights.size(); i++ )
         {
            vW[ i ].array() += dW[ i ].array().square();
            weights[ i ].array() -= rate * dW[ i ].array() / ( vW[ i ].array().sqrt() + 1e-7 );

            vB[ i ].array() += dB[ i ].array().square();
            biases[ i ].array() -= rate * dB[ i ].array() / ( vB[ i ].array().sqrt() + 1e-7 );
         }
      }
   };

   /// <summary>
   /// RMSprop
   /// </summary>
   class RmsProp : public BaseOptimizer
   {
      double rate;
      double decay;

      public:
      explicit RmsProp( double rate = 0.01, double decay = 0.99 )
         : rate( rate ), decay( decay )
      {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         Initialize( weights, biases );

         for( size_t i = 0; i < weights.size(); i++ )
         {
            // Weights decay in-place
            vW[ i ].array() = decay * vW[ i ].array() + ( 1.0 - decay ) * dW[ i ].array().square();
            weights[ i ].array() -= rate * dW[ i ].array() / ( vW[ i ].array().sqrt() + 1e-7 );

            // Biases decay in-place
            vB[ i ].array() = decay * vB[ i ].array() + ( 1.0 - decay ) * dB[ i ].array().square();
            biases[ i ].array() -= rate * dB[ i ].array() / ( vB[ i ].array().sqrt() + 1e-7 );
         }
      }
   };

   /// <summary>
   /// Adam
   /// </summary>
   class Adam : public BaseOptimizer
   {
      double rate;
      double beta1;
      double beta2;
      int iter;

      std::vector<MatrixXd> mW;
      std::vector<VectorXd> mB;

      public:
      explicit Adam( double rate = 0.001, double beta1 = 0.9, double beta2 = 0.999 )
         : rate( rate ), beta1( beta1 ), beta2( beta2 ), iter( 0 )
      {}

      void Update( std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases,
                   const std::vector<MatrixXd>& dW, const std::vector<VectorXd>& dB ) override
      {
         if( mW.empty() )
         {
            mW.resize( weights.size() );
            vW.resize( weights.size() );
            mB.resize( biases.size() );
            vB.resize( biases.size() );

            for( size_t i = 0; i < weights.size(); i++ )
            {
               mW[ i ] = MatrixXd::Zero( weights[ i ].rows(), weights[ i ].cols() );
               vW[ i ] = MatrixXd::Zero( weights[ i ].rows(), weights[ i ].cols() );
               mB[ i ] = VectorXd::Zero( biases[ i ].size() );
               vB[ i ] = VectorXd::Zero( biases[ i ].size() );
            }
         }

         iter++;
         double lr_t = rate * std::sqrt( 1.0 - std::pow( beta2, iter ) ) / ( 1.0 - std::pow( beta1, iter ) );

         for( size_t i = 0; i < weights.size(); i++ )
         {
            // Weights Update
            mW[ i ].array() = beta1 * mW[ i ].array() + ( 1.0 - beta1 ) * dW[ i ].array();
            vW[ i ].array() = beta2 * vW[ i ].array() + ( 1.0 - beta2 ) * dW[ i ].array().square();

            weights[ i ].array() -= lr_t * mW[ i ].array() / ( vW[ i ].array().sqrt() + 1e-7 );

            // Biases Update
            mB[ i ].array() = beta1 * mB[ i ].array() + ( 1.0 - beta1 ) * dB[ i ].array();
            vB[ i ].array() = beta2 * vB[ i ].array() + ( 1.0 - beta2 ) * dB[ i ].array().square();

            biases[ i ].array() -= lr_t * mB[ i ].array() / ( vB[ i ].array().sqrt() + 1e-7 );
         }
      }
   };
}