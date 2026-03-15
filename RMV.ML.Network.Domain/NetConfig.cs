using System;
using System.Configuration;

namespace RMV.Backpropagation
{
   /// <summary>
   /// Network configuration
   /// </summary>
	public class NetConfig : ConfigurationSection
	{
		static NetConfig section;

		static NetConfig()
		{
			section = ConfigurationManager.GetSection( "netConfig" ) as NetConfig;
		}

		public static int Input
		{
			get { return section.input; }
		}

		public static int[] Hidden
		{
			get { return Array.ConvertAll( section.hidden.Split( ',' ), a => int.Parse( a ) ); }
		}

		public static int Output
		{
			get { return section.output; }
		}

		public static double Eps
		{
			get { return section.eps; }
		}

		public static double Rate
		{
			get { return section.learning; }
		}

		//public static double Momentum
		//{
		//	get { return section.momentum; }
		//}

		public static int Iterations
		{
			get { return section.iterations; }
		}

      /// <summary>
      /// Batch size
      /// </summary>
      public static int Batch
      {
         get { return section.batch; }
      }

      /// <summary>
      /// Print interval
      /// </summary>
      public static int Print
      {
         get { return section.print; }
      }

      public static string Train
		{
			get { return section.train; }
		}

      public static string Test
      {
         get { return section.test; }
      }

      public static string Errors
      {
         get { return section.errors; }
      }

      [ConfigurationProperty( "input" )]
		int input
		{
			get { return ( int )this[ "input" ]; }
		}

		[ConfigurationProperty( "hidden" )]
		string hidden
		{
			get { return ( string ) this[ "hidden" ]; }
		}

		[ConfigurationProperty( "output" )]
		int output
		{
			get { return ( int )this[ "output" ]; }
		}

		[ConfigurationProperty( "eps" )]
		double eps
		{
			get { return ( double )this[ "eps" ]; }
		}

		[ConfigurationProperty( "learning" )]
		double learning
		{
			get { return ( double )this[ "learning" ]; }
		}

		[ConfigurationProperty( "momentum", IsRequired = false )]
		double momentum
		{
			get { return ( double )this[ "momentum" ]; }
		}

		[ConfigurationProperty( "iterations" )]
		int iterations
		{
			get { return ( int )this[ "iterations" ]; }
		}

      [ConfigurationProperty( "batch" )]
      int batch
      {
         get { return ( int )this[ "batch" ]; }
      }

      [ConfigurationProperty( "train" )]
		string train
		{
			get { return ( string )this[ "train" ]; }
		}

      [ConfigurationProperty( "test" )]
      string test
      {
         get { return ( string ) this[ "test" ]; }
      }

      [ConfigurationProperty( "errors" )]
      string errors
      {
         get { return ( string ) this[ "errors" ]; }
      }

      [ConfigurationProperty( "print" )]
      int print
      {
         get { return ( int ) this[ "print" ]; }
      }
   }
}
