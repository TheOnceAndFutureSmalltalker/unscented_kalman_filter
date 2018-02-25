#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  is_initialized_ = false;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  // initialize lambda
  lambda_ = 3 - n_aug_;

  // instantiate matrix for predicted sigma points
  Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1 );

  // initialize vector for weights - this remains constant throughout
  weights_ = VectorXd(2*n_aug_+1);
  double wt = 1 / (2*(lambda_ + n_aug_));
  weights_.fill(wt);
  weights_[0] = (lambda_ / (lambda_ + n_aug_));

  R_lidar = MatrixXd(2,2);
  R_lidar.fill(0.0);
  R_lidar(0,0) = std_laspx_*std_laspx_;
  R_lidar(1,1) = std_laspy_*std_laspy_;

  R_radar = MatrixXd(3,3);
  R_radar.fill(0.0);
  R_radar(0,0) = std_radr_*std_radr_;
  R_radar(1,1) = std_radphi_*std_radphi_;
  R_radar(2,2) = std_radrd_*std_radrd_;

  // DEGUB ONLY
  count = 0;
  print = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  //cout << endl;
  count++;

  if(is_initialized_)
  {
      // calculate delta_t and capture new state time
      double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // convert us to s

      // set a flag to print out data, this is where we cross x axis and things go wrong
      if(count > 272 && count < 277)
      {
          print = 1;
      }
      else
      {
        print = 0;
      }
      // update state from measurement
      // NOTE:  prediction and timestamp update are done within each conditional
      // in event we are only using one or the other type of device
      if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
      {
          // make prediction based on previous state
          Prediction(delta_t);
          // update from lidar measurement
          UpdateLidar(meas_package);
          // update timestamp
          time_us_ = meas_package.timestamp_;
      }
      if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
      {
          // make prediction based on previous state
          Prediction(delta_t);
          // update from radar measurement
          UpdateRadar(meas_package);
          // update timestamp
          time_us_ = meas_package.timestamp_;
      }
  }
  else
  {
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        // initialize from laser
        x_[0] = meas_package.raw_measurements_[0];
        x_[1] = meas_package.raw_measurements_[1];
        x_[2] = 0.0;
        if(x_[0] != 0.0)
        {
            x_[3] = atan2(x_[1], x_[0]);
        }
        else
        {
            x_[3] = 0.0;
        }
        x_[4] = 0.0;
    }
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        // initialize from radar
        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        double rho_dot = meas_package.raw_measurements_[2];

        x_[0] = rho * cos(phi);
        x_[1] = rho * sin(phi);
        x_[2] = 0.0;
        if(x_[0] != 0.0)
        {
            x_[3] = atan2(x_[1], x_[0]);
        }
        else
        {
            x_[3] = 0.0;
        }
        x_[4] = 0.0;
    }

    time_us_ = meas_package.timestamp_;

    P_.setIdentity();

    is_initialized_ = true;
  }
  //cout << x_ << endl;
  //cout << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //cout << "prediction step" << endl;

//if(print == 1)
//{
//    cout << endl << "STEP " << count << endl;
//    cout << "INIT " << x_[0] << " " << x_[1] << " " << x_[2] << " " << x_[3] << " " << x_[4] << endl;
//    //cout << endl << P_ << endl;
//}

  //create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0.0, 0.0;
  //cout << x_aug << endl;

  //create augmented covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.block(0,0,5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  //cout << P_aug << endl;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd root_lambda_nx_P = sqrt(lambda_ + n_aug_) * A;
  MatrixXd x_plus_root_lambda_nx_P = root_lambda_nx_P.colwise() + x_aug;
  MatrixXd x_minus_root_lambda_nx_P = (root_lambda_nx_P * -1.0).colwise() + x_aug;
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug << x_aug, x_plus_root_lambda_nx_P, x_minus_root_lambda_nx_P;
  //cout << Xsig_aug << endl;

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for (int i = 0, nCols = Xsig_aug.cols(); i < nCols; i++)
  {
      double px = Xsig_aug(0,i);
      double py = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double psi = Xsig_aug(3,i);
      double psi_dot = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_psi = Xsig_aug(6,i);

      if(psi_dot < 0.001) // if close to zero
      {
      Xsig_pred(0,i) = px + v*cos(psi)*delta_t + 0.5*delta_t*delta_t*cos(psi)*nu_a;
      Xsig_pred(1,i) = py + v*sin(psi)*delta_t + 0.5*delta_t*delta_t*sin(psi)*nu_a;
      }
      else
      {
      Xsig_pred(0,i) = px + (v/psi_dot)*(sin(psi+psi_dot*delta_t) - sin(psi))+ 0.5*delta_t*delta_t*cos(psi)*nu_a;
      Xsig_pred(1,i) = py + (v/psi_dot)*(-cos(psi+psi_dot*delta_t) + cos(psi)) + 0.5*delta_t*delta_t*sin(psi)*nu_a;
      }

      Xsig_pred(2,i) = v + 0.0 + delta_t*nu_a;
      Xsig_pred(3,i) = psi + psi_dot*delta_t + 0.5*delta_t*delta_t*nu_psi;
      Xsig_pred(4,i) = psi_dot + 0.0 + delta_t*nu_psi;

//      if(print == 1)
//      {
//          cout << "XSIG " << Xsig_pred(0,i) << " " << Xsig_pred(1,i) << " " << Xsig_pred(2,i) << " " << Xsig_pred(3,1) << endl;
//      }
  }
  //cout << Xsig_pred << endl;

  //predict state mean
  x_ = Xsig_pred * weights_;
  //cout << x_ << endl;

  // predict state covariance matrix
  MatrixXd error = Xsig_pred.colwise() - x_;
  for(int i=0; i<2*n_aug_+1;i++)
  {
      while(error(3,i)>M_PI) error(3,i)-=2.0*M_PI;
      while(error(3,i)<-M_PI) error(3,i)+=2.0*M_PI;
  }
  MatrixXd error_transposed = error.transpose();
  MatrixXd error_transposed_weighted = error_transposed.array().colwise() * weights_.array();
  P_ = error * error_transposed_weighted;

//if(print == 1)
//{
//    //cout << "PRED " << x_[0] << " " << x_[1] << " " << x_[2] << " " << x_[3] << " " << x_[4] << endl;
//    //cout << endl << P_ << endl;
//}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

//    if(print == 1)
//    {
//        cout << "LIDAR" << endl;
//    }
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  Zsig = Xsig_pred.block(0,0,n_z, 2 * n_aug_ + 1);
  //cout << Xsig_pred << endl;
  //cout << Zsig << endl;

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate innovation covariance matrix S
  MatrixXd error = Zsig.colwise() - z_pred;
  MatrixXd error_transposed = error.transpose();
  MatrixXd error_transposed_weighted = error_transposed.array().colwise() * weights_.array();
  S = error * error_transposed_weighted + R_lidar;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  MatrixXd X_error = Xsig_pred.colwise() - x_;
  MatrixXd Z_error = Zsig.colwise() - z_pred;
  MatrixXd Z_error_transposed = Z_error.transpose();
  MatrixXd Z_error_transposed_weighted = Z_error_transposed.array().colwise() * weights_.array();
  Tc = X_error * Z_error_transposed_weighted;
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  P_ = P_ - K * S * K.transpose();
  // calculate nis
  VectorXd diff = (meas_package.raw_measurements_ - z_pred);
  double nis = diff.transpose() * S.inverse() * diff;
  nis_radar.push_back(nis);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
//    if(print == 1)
//    {
//        cout << "RADAR" << endl;
//    }
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for(int i=0; i<2 * n_aug_ + 1; i++)
  {
      double px = Xsig_pred(0,i);
      double py = Xsig_pred(1,i);
      double v = Xsig_pred(2,i);
      double psi = Xsig_pred(3,i);
      double xy_mag = sqrt(px*px + py*py);
      Zsig(0,i) = xy_mag;
      if(px != 0.0)
      {
          Zsig(1,i) = atan2(py, px);
          // when we cross x axis and px is negative, py may be positive or negative
          // this means rho may be positive or negative, but around same magnitude, ~3.14
          // this yields a weighted average that is way off!!
          // if this happens, lets just bail out and wait for next measurement
          // note: need more robust solution here, but for, now this works!
          if(Zsig(1,i) < -3.12 || Zsig(1,i) > 3.12)
          {
              return;
          }
      }
      else
      {
          Zsig(1,i) = 0.0;
      }
      if(px*px + py*py < 0.00001)
      {
          Zsig(2,i) = 0.0;
      }
      else
      {
        Zsig(2,i) = (px*cos(psi)*v +py*sin(psi)*v) / xy_mag;
      }
  }

  // if some rho are positive and some are negative then the average will be way off
  // so check for this condition and set them all to 3.14



  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  // convert rho back to (-pi,pi)
  //if(z_pred(1)>M_PI) z_pred(1) -= 2*M_PI;
//  for(int i=0; i<2 * n_aug_ + 1; i++)
//  {
//      if(Zsig(1,i)>M_PI) Zsig(1,i) -= 2*M_PI;
//      if(print == 1)
//      {
//          cout << "ZSIG " << Zsig(0,i) << " " << Zsig(1,i) << " " << Zsig(2,i) << endl;
//      }
//  }


  if(print == 1)
  {
      cout << "ZPRD " << z_pred(0) << " " << z_pred(1) << " " << z_pred(2) << endl;
  }

  //calculate innovation covariance matrix S
  MatrixXd error = Zsig.colwise() - z_pred;
  for(int i=0; i<2*n_aug_+1;i++)
  {
      while(error(1,i)>M_PI) error(1,i)-=2.0*M_PI;
      while(error(1,i)<-M_PI) error(1,i)+=2.0*M_PI;
  }
  MatrixXd error_transposed = error.transpose();
  MatrixXd error_transposed_weighted = error_transposed.array().colwise() * weights_.array();
  S = error * error_transposed_weighted + R_radar;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  MatrixXd X_error = Xsig_pred.colwise() - x_;
  MatrixXd Z_error = Zsig.colwise() - z_pred;
  MatrixXd Z_error_transposed = Z_error.transpose();
  MatrixXd Z_error_transposed_weighted = Z_error_transposed.array().colwise() * weights_.array();
  Tc = X_error * Z_error_transposed_weighted;
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();


  //update state mean and covariance matrix
  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  P_ = P_ - K * S * K.transpose();
  // calculate nis
  VectorXd diff = (meas_package.raw_measurements_ - z_pred);
  double nis = diff.transpose() * S.inverse() * diff;
  nis_radar.push_back(nis);
}
