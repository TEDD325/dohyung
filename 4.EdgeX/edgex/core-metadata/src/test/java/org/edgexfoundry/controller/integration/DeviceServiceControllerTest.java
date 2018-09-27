/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.integration;

import static org.edgexfoundry.test.data.ServiceData.TEST_LABELS;
import static org.edgexfoundry.test.data.ServiceData.TEST_SERVICE_NAME;
import static org.edgexfoundry.test.data.ServiceData.checkTestData;
import static org.edgexfoundry.test.data.ServiceData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Set;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.DeviceServiceControllerImpl;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.AdminState;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.OperatingState;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class DeviceServiceControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  DeviceRepository deviceRepos;

  @Autowired
  DeviceProfileRepository profileRepos;

  @Autowired
  DeviceServiceControllerImpl controller;

  @Autowired
  DeviceServiceRepository repos;

  @Autowired
  private AddressableRepository addrRepos;

  private String id;
  private String addrId;

  @Before
  public void setup() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    addrId = addr.getId();
    DeviceService service = ServiceData.newTestInstance();
    service.setAddressable(addr);
    repos.save(service);
    id = service.getId();
  }

  @After
  public void cleanup() throws Exception {
    resetControllerMAXLIMIT();
    resetRepos();
    deviceRepos.deleteAll();
    profileRepos.deleteAll();
    addrRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testDeviceService() {
    DeviceService service = controller.deviceService(id);
    checkTestData(service, id);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServiceWithUnknownnId() {
    controller.deviceService("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceException() throws Exception {
    unsetRepos();
    controller.deviceService(id);
  }

  @Test
  public void testDeviceServices() {
    List<DeviceService> services = controller.deviceServices();
    assertEquals("Find all not returning a list with one device service", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesException() throws Exception {
    unsetRepos();
    controller.deviceServices();
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceServicesMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.deviceServices();
  }

  @Test
  public void testDeviceServiceForName() {
    DeviceService service = controller.deviceServiceForName(TEST_SERVICE_NAME);
    checkTestData(service, id);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServiceForNameWithNoneMatching() {
    controller.deviceServiceForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceForNameException() throws Exception {
    unsetRepos();
    controller.deviceServiceForName(TEST_SERVICE_NAME);
  }

  @Test
  public void testDeviceServiceByLabel() {
    List<DeviceService> services = controller.deviceServicesByLabel(TEST_LABELS[0]);
    assertEquals("Find for labels not returning appropriate list", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test
  public void testDeviceServicesByLabelWithNoneMatching() {
    assertTrue("No devices should be found with bad label",
        controller.deviceServicesByLabel("badlabel").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesByLabelException() throws Exception {
    unsetRepos();
    controller.deviceServicesByLabel(TEST_LABELS[0]);
  }

  @Test
  public void testDeviceServicesForAddress() {
    List<DeviceService> services = controller.deviceServicesForAddressable(addrId);
    assertEquals("Find for address not returning appropriate list", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test
  public void testDeviceServicesForAddressableByName() {
    List<DeviceService> services =
        controller.deviceServicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
    assertEquals("Find for address not returning appropriate list", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServicesForAddressWithNoneMatching() throws Exception {
    controller.deviceServicesForAddressable("badaddress");
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServicesForAddressByNameWithNoneMatching() throws Exception {
    controller.deviceServicesForAddressableByName("badaddress");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesForAddressException() throws Exception {
    unsetRepos();
    controller.deviceServicesForAddressable(addrId);
  }

  @Test
  public void testAddressablesForAssociatedDevices() {
    DeviceService service = repos.findOne(id);
    Device device = DeviceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    profileRepos.save(profile);
    device.setProfile(profile);
    device.setService(service);
    Addressable addressable = addrRepos.findOne(addrId);
    device.setAddressable(addressable);
    deviceRepos.save(device);
    Set<Addressable> addressables = controller.addressablesForAssociatedDevices(id);
    assertEquals(
        "Find addressables for associated devices not returning appropriate list of addressable", 1,
        addressables.size());
    AddressableData.checkTestData((Addressable) addressables.toArray()[0], addrId);
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableForAssociatedDevicesWithBadService() {
    controller.addressablesForAssociatedDevices("badserviceid");
  }

  @Test
  public void testAddressableForAssociatedDevicesByName() {
    DeviceService service = repos.findOne(id);
    Device device = DeviceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    profileRepos.save(profile);
    device.setProfile(profile);
    device.setService(service);
    Addressable addressable = addrRepos.findOne(addrId);
    device.setAddressable(addressable);
    deviceRepos.save(device);
    Set<Addressable> addressables =
        controller.addressablesForAssociatedDevicesByName(TEST_SERVICE_NAME);
    assertEquals(
        "Find addressables for associated devices by name not returning appropriate list of addressable",
        1, addressables.size());
    AddressableData.checkTestData((Addressable) addressables.toArray()[0], addrId);
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableForAssociatedDevicesByNameWithBadService() {
    controller.addressablesForAssociatedDevicesByName("badservicename");
  }

  @Test
  public void testAdd() {
    DeviceService service = repos.findOne(id);
    service.setId(null);
    service.setName("NewName");
    String newId = controller.add(service);
    assertNotNull("New device id is null", newId);
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testAddNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    DeviceService service = newTestInstance();
    controller.add(service);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNoAddressable() {
    DeviceService service = repos.findOne(id);
    service.setId(null);
    service.setName("newname");
    service.setAddressable(null);
    controller.add(service);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    DeviceService service = repos.findOne(id);
    service.setId(null);
    service.setName("NewName");
    controller.add(service);
  }

  @Test
  public void testDelete() {
    assertTrue("Delete did not return correctly", controller.delete(id));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteWithNone() {
    controller.delete("badid");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() throws Exception {
    unsetRepos();
    controller.delete(id);
  }

  @Test
  public void testDeleteByName() {
    assertTrue("Delete did not return correctly", controller.deleteByName(TEST_SERVICE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(TEST_SERVICE_NAME);
  }

  @Test
  public void testUpdate() {
    DeviceService service = repos.findOne(id);
    service.setDescription("newdescription");
    assertTrue("Update did not complete successfully", controller.update(service));
    DeviceService service2 = repos.findOne(id);
    assertEquals("Update did not work correclty", "newdescription", service2.getDescription());
    assertNotNull("Modified date is null", service2.getModified());
    assertNotNull("Create date is null", service2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service2.getModified() != service2.getCreated());
  }

  @Test
  public void testUpdateLastConnected() {
    assertTrue("Update did not complete successfully", controller.updateLastConnected(id, 1000));
    DeviceService service = repos.findOne(id);
    assertEquals("Update last connected did not work correclty", 1000, service.getLastConnected());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastConnectedNoneFound() {
    controller.updateLastConnected("badid", 1000);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedException() throws Exception {
    unsetRepos();
    controller.updateLastConnected(id, 1000);
  }

  @Test
  public void testUpdateLastConnectedByName() {
    assertTrue("Update did not complete successfully",
        controller.updateLastConnectedByName(TEST_SERVICE_NAME, 1000));
    DeviceService service = repos.findByName(TEST_SERVICE_NAME);
    assertEquals("Update last connected did not work correclty", 1000, service.getLastConnected());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastConnectedByNameNoneFound() {
    controller.updateLastConnectedByName("badname", 1000);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedByNameException() throws Exception {
    unsetRepos();
    controller.updateLastConnectedByName(TEST_SERVICE_NAME, 1000);
  }

  @Test
  public void testUpdateLastReported() {
    assertTrue("Update did not complete successfully", controller.updateLastReported(id, 1000));
    DeviceService service = repos.findOne(id);
    assertEquals("Update last reported did not work correclty", 1000, service.getLastReported());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastReportedNoneFound() {
    controller.updateLastReported("badid", 1000);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedException() throws Exception {
    unsetRepos();
    controller.updateLastReported(id, 1000);
  }

  @Test
  public void testUpdateLastReportedByName() {
    assertTrue("Update did not complete successfully",
        controller.updateLastReportedByName(TEST_SERVICE_NAME, 1000));
    DeviceService service = repos.findByName(TEST_SERVICE_NAME);
    assertEquals("Update last reported did not work correclty", 1000, service.getLastReported());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastReportedByNameNoneFound() {
    controller.updateLastReportedByName("badname", 1000);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedByNameException() throws Exception {
    unsetRepos();
    controller.updateLastReportedByName(TEST_SERVICE_NAME, 1000);
  }

  @Test
  public void testUpdateOpState() {
    assertTrue("Update did not complete successfully",
        controller.updateOpState(id, OperatingState.DISABLED.toString()));
    DeviceService service = repos.findOne(id);
    assertEquals("Update op state did not work correclty", OperatingState.DISABLED,
        service.getOperatingState());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateNoneFound() {
    controller.updateOpState("badid", OperatingState.DISABLED.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateException() throws Exception {
    unsetRepos();
    controller.updateOpState(id, OperatingState.DISABLED.toString());
  }

  @Test
  public void testUpdateOpStateByName() {
    assertTrue("Update did not complete successfully",
        controller.updateOpStateByName(TEST_SERVICE_NAME, OperatingState.DISABLED.toString()));
    DeviceService service = repos.findByName(TEST_SERVICE_NAME);
    assertEquals("Update op state did not work correclty", OperatingState.DISABLED,
        service.getOperatingState());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateByNameNoneFound() {
    controller.updateOpStateByName("badname", OperatingState.DISABLED.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateByNameException() throws Exception {
    unsetRepos();
    controller.updateOpStateByName(TEST_SERVICE_NAME, OperatingState.DISABLED.toString());
  }

  @Test
  public void testUpdateAdminState() {
    assertTrue("Update did not complete successfully",
        controller.updateAdminState(id, AdminState.LOCKED.toString()));
    DeviceService service = repos.findOne(id);
    assertEquals("Update admin state did not work correclty", AdminState.LOCKED,
        service.getAdminState());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateAdminStateNoneFound() {
    controller.updateAdminState("badid", AdminState.LOCKED.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateException() throws Exception {
    unsetRepos();
    controller.updateAdminState(id, AdminState.LOCKED.toString());
  }

  @Test
  public void testUpdateAdminStateByName() {
    assertTrue("Update did not complete successfully",
        controller.updateAdminStateByName(TEST_SERVICE_NAME, AdminState.LOCKED.toString()));
    DeviceService service = repos.findByName(TEST_SERVICE_NAME);
    assertEquals("Update admin state did not work correclty", AdminState.LOCKED,
        service.getAdminState());
    assertNotNull("Modified date is null", service.getModified());
    assertNotNull("Create date is null", service.getCreated());
    assertTrue("Modified date and create date should be different after update",
        service.getModified() != service.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateAdminStateByNameNoneFound() {
    controller.updateOpStateByName("badname", AdminState.LOCKED.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateByNameException() throws Exception {
    unsetRepos();
    controller.updateOpStateByName(TEST_SERVICE_NAME, AdminState.LOCKED.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    DeviceService service = repos.findOne(id);
    service.setDescription("newdescription");
    controller.update(service);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithBadAddressable() {
    DeviceService service = repos.findOne(id);
    Addressable addr = AddressableData.newTestInstance();
    addr.setName("baddaddressname");
    service.setAddressable(addr);
    controller.update(service);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    DeviceService service = repos.findOne(id);
    service.setId("badid");
    service.setName("badname");
    service.setDescription("newdescription");
    controller.update(service);
  }

  private void unsetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, repos);
  }

  private void unsetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 0);
  }

  private void resetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 1000);
  }
}
