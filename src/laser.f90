module laser
  use global
  implicit none

  PRIVATE

  public :: irradiance, power, W

contains
  pure function irradiance(rad, t, z)
    real(dp), intent(in) :: rad, t, z
    real(dp) :: irradiance, phi, w_z
    w_z = W(z)
    phi = power(t)
    irradiance = 0.0001*(2*phi)/(PI_dp*w_z**2)*dexp(-2*rad**2/w_z**2)
  end function irradiance

  pure function power(t)
    real(dp), intent(in) :: t
    real(dp) :: power
    power = (0.94_dp * ep / tau) * dexp(-(t / wid)**2)
  end function power

  pure function W(z)
    real(dp), intent(in) :: z
    real(dp) :: W
    W = w0 * sqrt(1 + (z / zr)**2)
  end function W

end module laser

