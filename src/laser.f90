module laser
  use global
  implicit none

  PRIVATE

  public :: irradiance, power, W

contains
  pure function irradiance(rad, t, z, ep, wid, w0, zr)
    real(c_double), intent(in) :: rad, t, z, ep, wid, w0, zr
    real(c_double) :: irradiance, phi, w_z
    w_z = W(z, w0, zr)
    phi = power(t, ep, wid)
    irradiance = 0.0001_c_double * (2 * phi) / (pi * w_z**2) * dexp(-2 * rad**2 / w_z**2)
  end function irradiance

  pure function power(t, ep, wid)
    real(c_double), intent(in) :: t, ep, wid
    real(c_double) :: power
    power = (0.94_c_double * ep / wid) * dexp(-(t / wid)**2)
  end function power

  pure function W(z, w0, zr)
    real(c_double), intent(in) :: z, w0, zr
    real(c_double) :: W
    W = w0 * sqrt(1 + (z / zr)**2)
  end function W

end module laser
