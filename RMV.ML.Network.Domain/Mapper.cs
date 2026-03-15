using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace RMV.Backpropagation
{
	/// <summary>
	/// Maps data to the Activation Function range
	/// </summary>
	public class Mapper
	{				
		public IActivation Activation { get; set; }
     

      //public double[] Normalize( IEnumerable<double> data )
      //{
      //   double min = data.Min();
      //   double max = data.Max();
      //   var scale = max - min;
      //   var delta = this.activation.Max - this.activation.Min;			
      //   return data.Select( i => ( i - min ) * delta / scale + this.activation.Min ).ToArray();		
      //}

      /// <summary>
      /// Normalizes data using Min-Max scaling
      /// </summary>
      /// <param name="data">target data</param>
      /// <returns>normalized data</returns>
      public IEnumerable<double> NormalizeMinMax( IEnumerable<double> data )
      {
         double min = data.Min();
         double max = data.Max();

         var delta = this.Activation.Max - this.Activation.Min;

         return data.Select( d => ( d - min ) * delta / ( max - min ) + this.Activation.Min );
      }

		/// <summary>
		/// Normalizes data using Z-score standardization
		/// </summary>
		/// <param name="data">target data</param>
		/// <returns>normalized data</returns>
		public IEnumerable<double> NormalizeZscore( IEnumerable<double> data )
      {
         double mean = data.Average();
         double stdev = data.

         return data.Select( d => ( d - min ) * delta / ( max - min ) + this.Activation.Min );
      }

		/// <summary>
		/// Normalizes data using a custom scaling method based on the Activation Function range
		/// </summary>      
		/// <param name="data">target data</param>
		/// <returns>normalized data</returns>
		public IEnumerable<double> Normalize( IEnumerable<double> data )
      {
         var delta = this.Activation.Max - this.Activation.Min;

         return data.Select( i => ( i - this.Min ) * delta / this.Scale + this.Activation.Min );
      }

      /// <summary>
      /// Denormalizes data for display
      /// </summary>
      /// <param name="data">data</param>
      /// <param name="max">max value</param>
      /// <param name="min">min value</param>
      /// <returns>denormalized data</returns>
      public IEnumerable<double> Denormalize( IEnumerable<double> data, double max, double min  )
		{
			var scale = max - min;

			var delta = this.Activation.Max - this.Activation.Min;

         return data.Select( i => ( i - this.Activation.Min ) * scale / delta + min );		
		}
	}
}
