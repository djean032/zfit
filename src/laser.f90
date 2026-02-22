module laser
  use global
  implicit none

  private

  public :: irradiance

contains
  pure function irradiance(rad, t, z, ep, wid, w0, zr)
    real(c_double), intent(in) :: rad, t, z, ep, wid, w0, zr
    real(c_double) :: irradiance, phi, w_z
    w_z = w0 * sqrt(1 + (z/zr)**2)
    phi = (0.94_c_double*ep/wid) * dexp(-(t/wid)**2)
    irradiance = 0.0001_c_double * (2*phi) / (pi*w_z**2) * dexp(-2*rad**2/w_z**2)
  end function irradiance

end module laser
