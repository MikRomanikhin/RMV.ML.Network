using MathNet.Numerics.LinearAlgebra;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Interface for optimization algorithms used in training neural networks
/// </summary>
interface IOptimizer
{
   void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads );
}

//public abstract class BaseOptimizer
//{	
//	public BaseOptimizer( double rate )
//	{
//		this.rate = rate;
//	}
//	protected double rate;

//	public abstract void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads );
//}

/// <summary>
/// Stochastic Gradient Descent
/// </summary>
public class SGD( double rate = 0.01 ) : IOptimizer
{
   public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
   {
      foreach( var key in parameters.Keys )
      {
         // In-place, zero-allocation update: param = param - (rate * grad)
         parameters[ key ].Map2( ( p, g ) => p - ( rate * g ), grads[ key ], parameters[ key ] );
      }
   }
}


/// <summary>
/// Momentum SGD
/// </summary>
public class Momentum( double rate = 0.01, double momentum = 0.9 ) : IOptimizer
{
	Dictionary<string, Matrix<double>>? v = null;

	public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
	{
		if( v == null )
		{
			v = [];
			foreach( var kvp in parameters )
			{
				v[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
			}
		}

		foreach( var key in parameters.Keys )
		{
			var g = grads[ key ];	var vk = v[ key ];	var p = parameters[ key ];
			
			vk.Map2( ( vi, gi ) => momentum * vi - rate * gi, g, vk ); // v = μ * v - lr * g — in-place
			
			p.Map2( ( pi, vi ) => pi + vi, vk, p ); // p += v — in-place
		}
	}
}

/// <summary>
/// Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
/// </summary>
public class Nesterov( double rate = 0.01, double momentum = 0.9 ) : IOptimizer
{
	Dictionary<string, Matrix<double>>? v = null;

	public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
	{
		if( v == null )
		{
			v = [];
			foreach( var kvp in parameters )
			{
				v[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
			}
		}

		foreach( var key in parameters.Keys )
		{
			var g = grads[ key ];	var vk = v[ key ];	var p = parameters[ key ];
						
			var vPrev = vk.Clone(); // v_prev = v (snapshot before update)
						
			vk.Map2( ( vi, gi ) => momentum * vi - rate * gi, g, vk ); // v = μ * v - lr * g — in-place

			// p = p - μ * v_prev + (1 + μ) * v  — Nesterov lookahead
			// Equivalent to: p += -μ * v_prev + (1 + μ) * v
			p.Map2( ( pi, vpi ) => pi - momentum * vpi, vPrev, p );
			p.Map2( ( pi, vi ) => pi + ( 1 + momentum ) * vi, vk, p );
		}
	}
}

/// <summary>
/// AdaGrad (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
/// </summary>
public class AdaGrad( double rate = 0.01 ) : IOptimizer
{
	Dictionary<string, Matrix<double>>? h = null;

	public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
	{
		if( h == null )
		{
			h = [];
			foreach( var kvp in parameters )
			{
				h[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
			}
		}

		foreach( var key in parameters.Keys )
		{
			var g = grads[ key ];	var hk = h[ key ];	var p = parameters[ key ];
         			
			hk.Map2( ( hi, gi ) => hi + gi * gi, g, hk ); // h += g² — in-place accumulation

			// p = p - rate * g / (sqrt(h) + eps) — 1 temp allocation for step
			var step = g.Map2( ( gi, hi ) => rate * gi / ( Math.Sqrt( hi ) + 1e-7 ), hk );
         			
			p.Map2( ( pi, si ) => pi - si, step, p ); // p -= step — in-place
		}
	}
}

/// <summary>
/// RMSprop (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
/// </summary>
public class RmsProp( double rate = 0.01, double decay = 0.99 ) : IOptimizer
{   
   Dictionary<string, Matrix<double>>? h;

   public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
   {
      if( h == null )
      {
         h = [];
         foreach( var kvp in parameters )
         {
            h[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
         }
      }

      foreach( var key in parameters.Keys )
      {
         h[ key ] *= decay;
         h[ key ] += ( 1d - decay ) * grads[ key ].PointwiseMultiply( grads[ key ] );

         var hSqrt = h[ key ].Map( x => Math.Sqrt( x ) + 1e-7 );
         var step = grads[ key ].PointwiseDivide( hSqrt ) * rate;

         parameters[ key ] -= step;
      }
   }
}

/// <summary>
/// Adam (http://arxiv.org/abs/1412.6980v8)
/// </summary>
public class Adam( double rate = 0.001, double beta1 = 0.9, double beta2 = 0.999 ) : IOptimizer
{
	int iter = 0;
	Dictionary<string, Matrix<double>>? m, v;

	public void Update( Dictionary<string, Matrix<double>> parameters, Dictionary<string, Matrix<double>> grads )
	{
		if( m == null )
		{
			m = []; v = [];

			foreach( var kvp in parameters )
			{
				m[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
				v[ kvp.Key ] = Matrix<double>.Build.Dense( kvp.Value.RowCount, kvp.Value.ColumnCount, 0.0 );
			}
		}

		iter++;
		double lr_t = rate * Math.Sqrt( 1.0 - Math.Pow( beta2, iter ) ) / ( 1.0 - Math.Pow( beta1, iter ) );

		foreach( var key in parameters.Keys )
		{
			var g = grads[ key ];
			var mk = m[ key ];
			var vk = v[ key ];
			var p = parameters[ key ];

			// m = beta1 * m + (1 - beta1) * g  — in-place via result parameter
			mk.Map2( ( mi, gi ) => beta1 * mi + ( 1 - beta1 ) * gi, g, mk );

			// v = beta2 * v + (1 - beta2) * g²  — in-place via result parameter
			vk.Map2( ( vi, gi ) => beta2 * vi + ( 1 - beta2 ) * gi * gi, g, vk );

			// step = m / (sqrt(v) + eps) — in-place into a temp derived from mk
			var step = mk.Map2( ( mi, vi ) => lr_t * mi / ( Math.Sqrt( vi ) + 1e-7 ), vk );
         			
			p.Map2( ( pi, si ) => pi - si, step, p ); // p = p - step — in-place via result parameter
		}
	}
}
