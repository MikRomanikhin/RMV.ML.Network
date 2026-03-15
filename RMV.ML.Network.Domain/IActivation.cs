namespace RMV.ML.Network.Domain;

#region IActivation -----------------------------------------------------------

public interface IActivation
{
	/// <summary>
	/// Activation Function
	/// </summary>
	/// <param name="x">value which needs to be activated</param>	
	double Function( double x );

	/// <summary>
	/// Derivative of the Activation function
	/// </summary>	
	double Derivative( double x );

	/// <summary>
	/// Min value (for nomalization)
	/// </summary>
	double Min { get; }

	/// <summary>
	/// Max value (for nomalization)
	/// </summary>
	double Max { get;}	
}

#endregion


#region Identity --------------------------------------------------------------

/// <summary>
/// Identity activation (pass-through, used with Softmax output layer)
/// </summary>
[Serializable]
public class Identity : IActivation
{
	/// <summary>
	/// Activation function (identity: returns input unchanged).
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => x;

	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>
	public double Derivative( double x ) => 1d;

	public double Min => double.MinValue;

	public double Max => double.MaxValue;
}

#endregion


#region Step ---------------------------------------------------------------

[Serializable]
public class Step : IActivation
{
	/// <summary>
	/// Activation function.
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => x > 0 ? 1d : -1d;

	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>
	public double Derivative( double x ) => 1d;

	public double Min => -1d;

	public double Max => 1d;
}

#endregion


#region Relu ---------------------------------------------------------------

[Serializable]
public class Relu : IActivation
{
	/// <summary>
	/// Activation function.
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => Math.Max( x, 0 );

	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>
	public double Derivative( double x ) => x > 0 ? 1 : 0;

	public double Min => 0;

	public double Max => 1;
}

#endregion


#region Sigmoid -------------------------------------------------------

/// <summary>
/// Sigmoid activation
/// </summary>
[Serializable]
public class Sigmoid : IActivation
{
	/// <summary>
	/// Activation function.
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => 1d / ( Math.Exp( -x ) + 1d );


	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>	
	public double Derivative( double x ) => x * ( 1d - x );

	public double Min => 0;

	public double Max => 1;
}

#endregion


#region Bipolar -------------------------------------------------------

/// <summary>
/// Bipolar activation
/// </summary>
[Serializable]
public class Bipolar : IActivation
{
	/// <summary>
	/// Activation function.
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => 2d / ( 1d + Math.Exp( -x ) ) - 1d;	
		//double tmp = Math.Exp( -x );
		//return ( 1d - tmp ) / ( 1d + tmp );
		//return Math.Tanh( x );	
	

	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>	
	public double Derivative( double x )
	{
		double tmp = Function( x );

		return 0.5 * ( 1d - tmp * tmp ); //( 1 - tmp )*( 1 + tmp );
	}

	public double Min => -1d;

	public double Max => 1d;
}

#endregion


#region Hyperbolic tangent --------------------------------------------

/// <summary>
/// Hyperbolic tangent activation
/// </summary>
[Serializable]
public class Tanh : IActivation
{
	/// <summary>
	/// Activation function.
	/// </summary>
	/// <param name="x">value to be activated</param>	
	public double Function( double x ) => Math.Tanh( x );

	/// <summary>
	/// Derivative of the Activation function.
	/// </summary>	
	public double Derivative( double x )
	{
		double tmp = Function( x );

		return 1d - tmp * tmp;
	}

	public double Min => -1d;

	public double Max => 1d;
}

#endregion


   #region Softmax -------------------------------------------------------

   //public class SoftMax
   //{
   //   public void Function( double[] x )
   //   {
   //      double sum = x.Sum( i => Math.Exp( i ) );

   //      Array.ForEach( x, i => i = i / sum );
   //   }

   //   public double Derivative( double x )
   //   {
   //      return x * ( 1.0 - x );
   //   }

   //   public double Min
   //   {
   //      get { return 0; }
   //   }

   //   public double Max
   //   {
   //      get { return 1; }
   //   }
   //}

   #endregion
